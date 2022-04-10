import math
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable
from temporal.tableaux import *

torch.set_default_dtype(torch.float64)

class Solver(nn.Module):
    """
    Generic PDE solver
    """
    def __init__(self, time_solver: Any, spatial_solver: Any) -> None:
        super().__init__()
        self.time_solver = time_solver
        self.spatial_solver = spatial_solver

    def __str__(self):
        return str(self.time_solver)

    def solve(self, x0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Solve with initial conditions x0
        Args:
            x0 (torch.Tensor): initial conditions
            times (torch.Tensor): times
        Returns:
            torch.Tensor: [batch, num_steps, ...] solutions
        """
        u = [x0]
        dtimes = times[:, 1:] - times[:, :-1]
        self.time_solver.flush()
        for i in range(times.shape[1] - 1):
            dt, t = dtimes[:, i], times[:, i]
            update = self.time_solver.step(dt, self.spatial_solver, u[-1], t)
            u.append(update)
        return torch.stack(u, 1)


def time_solver(solver: Any, u0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Solve f with initial conditions u0
    Args:
        solver (Any):
        u0 (torch.Tensor): initial conditions
        times (torch.Tensor): times
    Returns:
        torch.Tensor: [batch, num_steps, ...] solutions
    """
    solver.flush()
    u = [u0]
    for i in range(times.shape[1] - 1):
        dt, t = times[:, i+1] - times[:, i], times[:, i]
        u.append(solver.step(dt, t, u[-1]))
    return torch.stack(u, 1)


class Iterator(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return ""

    def flush(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def _push_to_queue(self, a: torch.Tensor, b: torch.Tensor, buf: int=1) -> torch.Tensor:
        """Concatenate two tensors, where one is a buffer
         + b --> c
        a.shape = [s0,...,sn-1,sn]
        b.shape = [s0,...,sn-1]
        c.shape = [s0,...,sn-1,sn+1-buf]
        Args:
            a (torch.Tensor/None): [shape, n] buffer tensor
            b (torch.Tensor): [shape] tensor to append
            buf (int): number entries of tensor to remove when concatenating
        Returns:
            Concatenated tensors, unless a is None, then return b with extra last dim
        """
        if (a is None) and (b is not None):
            return b[..., None]
        else:
            return torch.cat([a[..., buf:], b[..., None]], -1)


class RKSolver(Iterator):
    """An inplicit Runge-Kutta Solver"""
    def __init__(self, tableau: Tableau, device: torch.cuda.device="cpu", conserve=False) -> None:
        """Instantiate RK solver
        Args:
            tableau (Tableau): an explicit RK tableau
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()
        self.tableau = tableau
        self.device = device
        if tableau.is_explicit:
            self.method = ERKSolver(tableau, device, conserve=conserve)
        else:
            NotImplementedError

    def __str__(self):
        if self.tableau.is_explicit:
            return str(self.tableau)
        else:
            return str(self.tableau)

    def step(self, h: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], yin: torch.Tensor, tin:torch.Tensor) -> torch.Tensor:
        """Compute one step of an RK solver
        Args:
            h (torch.Tensor): time increment
            f (Callable[[torch.Tensor], torch.Tensor]: function handle of form f(time, current-solution)
            tin (torch.Tensor): time at start of step
            yin (torch.Tensor): tensor of current solution
        Returns:
            torch.Tensor: new solution at time tin + h
        """
        tin = tin.to(self.device)
        return self.method.step(h, f, yin, tin)


class ERKSolver(Iterator):
    """An explicit Runge-Kutta Solver"""
    def __init__(self, tableau: Tableau, device: torch.cuda.device="cpu", conserve=False) -> None:
        """Instantiate ERK solver
        Args:
            tableau (Tableau): an explicit RK tableau
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()
        self.tableau = tableau
        self.device = device
        self.a = torch.Tensor(tableau.a).to(self.device)
        self.b = torch.Tensor(tableau.b).to(self.device)
        self.c = torch.Tensor(tableau.c).to(self.device)
        self.s = tableau.s
        assert tableau.is_explicit, "Explicit tableau expected"
        self.conserve = conserve
        self.adaptive = tableau.is_adaptive
        if self.adaptive:
            self.blo = torch.Tensor(tableau.blo).to(self.device)
            self.atol = torch.tensor(tableau.atol).to(self.device)
            self.rtol = torch.tensor(tableau.rtol).to(self.device)

    def __str__(self):
        return "ERK: tableau {:s}".format(str(self.tableau))

    def step(self, h, f, yin, tin):
        """Compute one step of an RK solver
        Args:
            h (torch.Tensor): time increment
            f (Callable[[torch.Tensor], torch.Tensor]: function handle of form f(time, current-solution)
            tin (torch.Tensor): time at start of step
            yin (torch.Tensor): tensor of current solution
        Returns:
            torch.Tensor: new solution at time tin + h
        """

        k = None
        s = None
        w = None

        h_tile = h
        h_tile = h_tile.to(self.device)
        for __ in range(len(yin.shape) - 1):
            h_tile = h_tile.unsqueeze(-1)
        # Compute intermediate stages
        for stage in range(self.s):
            if stage > 0:
                # Evaluate successive derivatives using RK rules
                teval = tin + h * self.c[stage]
                yeval = yin + h_tile * (k @ self.a[stage, :stage])
            else:
                teval = tin
                yeval = yin

            # Forward pass of spatial solver
            ki = f(teval, yeval)
            k = self._push_to_queue(k, ki, buf=0)

        # Linearly predict result from intermediate stages
        if self.conserve:
            k = k - torch.mean(k, -2, keepdim=True)
        if self.adaptive:
            y_hi = yin + h_tile * (k @ self.b)
            y_lo = yin + h_tile * (k @ self.blo)

            ymax, __ = torch.max(torch.max(torch.abs(yin), torch.abs(y_hi)), 0, keepdim=True)
            sc = self.atol + ymax * self.rtol

            error = torch.sqrt(torch.mean(((y_hi - y_lo) / sc)**2, -1))
            error = torch.max(error)

            if error >= 1.:
                hnew = 0.5 * h
                ymid = self.step(hnew, f, yin, tin)
                return self.step(hnew, f, ymid, tin + 0.5 * h)
            else:
                return y_hi
        else:
            return yin + h_tile * (k @ self.b)





