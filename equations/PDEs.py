import os
import sys
import math
import numpy as np
import torch
from torch import nn
from common.derivatives import WENO, FDM
from temporal.solvers import *


class PDE(nn.Module):
    """Generic PDE template"""
    def __init__(self):
        # Data params for grid and initial conditions
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def FDM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite differences method template"""
        pass

    def FVM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite volumes method template"""
        pass

    def WENO_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A WENO reconstruction template"""
        pass


class CE(PDE):
    """
    Combined equation with Burgers and KdV as edge cases
    ut = -alpha*uux + beta*uxx + -gamma*uxxx = 0
    alpha = 6 for KdV and alpha = 1. for Burgers
    beta = nu for Burgers
    gamma = 1 for KdV
    alpha = 0, beta = nu, gamma = 0 for heat equation
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 flux_splitting: str=None,
                 alpha: float=3.,
                 beta: float=0.,
                 gamma: float=1.,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            flux_splitting (str): flux splitting used for WENO reconstruction (Godunov, Lax-Friedrichs)
            alpha (float): shock term
            beta (float): viscosity/diffusion parameter
            gamma (float): dispersive parameter
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 0.5 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 3
        # Number of different waves
        self.N = 5
        # Length of the spatial domain / periodicity
        self.L = 16 if L is None else L
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # Initialize WENO reconstrution object
        self.weno = WENO(self, order=3, device=self.device)
        self.fdm = FDM(self, device=self.device)
        self.force = None
        self.flux_splitting = f'godunov' if flux_splitting is None else flux_splitting

        assert (self.flux_splitting == f'godunov') or (self.flux_splitting == f'laxfriedrichs')


    def __repr__(self):
        return f'CE'

    def flux(self, input: torch.Tensor) -> torch.Tensor:
        """
        Flux as used in weno scheme for CE equations
        """
        return 0.5 * input ** 2

    def FDM_reconstruction(self, t: float, u):
        raise


    def WENO_reconstruction(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives using WENO scheme
        update = -alpha*uux + beta*uxx - gamma*uxxx
        weno reconstruction for uux
        FDM reconstruction gives uxx, uxxx
        Args:
            t (torch.Tensor): timepoint at which spatial terms are reconstructed, only important for time-dependent forcing term
            u (torch.Tensor): input fields at given timepoint
        Returns:
            torch.Tensor: reconstructed spatial derivatives
        """
        dudt = torch.zeros_like(u)

        # WENO reconstruction of advection term
        u_padded_weno = self.weno.pad(u)
        if self.flux_splitting == f'godunov':
            dudt = - self.alpha * self.weno.reconstruct_godunov(u_padded_weno, self.dx)

        if self.flux_splitting == f'laxfriedrichs':
            dudt = - self.alpha * self.weno.reconstruct_laxfriedrichs(u_padded_weno, self.dx)

        # reconstruction of diffusion term
        u_padded_fdm = self.fdm.pad(u)
        uxx = self.fdm.second_derivative(u_padded_fdm)
        uxxx = self.fdm.third_derivative(u_padded_fdm)

        dudt += self.beta*uxx
        dudt -= self.gamma*uxxx

        # Forcing term
        if self.force:
            dudt += self.force(t)

        return dudt


class WE(PDE):
    """
    utt = c2uxx
    Dirichlet BCs:  u(−1,t)=0  and  u(+1,t)=0  for all  t>0
    Neumann BCs:  ux(−1,t)=0 and  ux(+1,t)=0  for all  t>0
    We implement the 2nd-order in time PDE as a 1st-order augmented state-space equation.
    We introduce a new variable  v, such that  ut=vut=v , so  ut=v, so utt=vt.
    For discretization, it is better to just use v as a storage variable for ut and compute utt directly from u.
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 xmin: float=None,
                 xmax: float=None,
                 grid_size: list=None,
                 bc_left: str=None,
                 bc_right: str=None,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            xmin (float): left spatial boundary
            xmax (float): right spatial boundary
            grid_size (list): grid points [nt, nx]
            bc_left (str): left boundary condition [Dirichlet, Neumann]
            bc_right (str): right boundary condition [Dirichlet, Neumann]
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20 if tmax is None else tmax
        # Left and right spatial boundaries
        self.xmin = -8 if xmin is None else xmin
        self.xmax = 8 if xmax is None else xmax
        # Length of the spatial domain
        self.L = abs(self.xmax - self.xmin)
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        self.dt = self.tmax / (self.grid_size[0] - 1)
        self.dx = self.L / (self.grid_size[1] - 1)
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        self.bc_left = "dirichlet" if bc_left is None else bc_left
        self.bc_right = "dirichlet" if bc_right is None else bc_right
        # Initialize Chebyshev pseudospectral solvers
        self.cheb = Cheb()

    def __repr__(self):
        return f'WE'

    def chebdx(self, t: np.array, u: np.array, x: np.array, c: float = 1.) -> np.array:
        """
        Compute the spatial derivatives using the pseudo-spectral method.
        Args:
            t (np.array): timepoint at which spatial terms are reconstructed, only important for time-dependent forcing term
            x (np.array): non-regular spatial grid
            u (np.array): input fields at given timepoint
            c (float): wave speed
        Returns:
            np.array: reconstructed spatial derivatives
        """
        N = len(u)
        # We store the first derivative in v and 2nd derivative in w. We do not use an extra
        # array dimension, because solve_ivp only works with 1D arrays.
        v, w = u[:N // 2], u[N // 2:]

        # Compute
        vt = w

        # The BCs are implemented with the dictionary. The key is the spatial derivative order,
        # the left and right values are the boundary condition values
        boundary_condition_left = 0
        boundary_condition_right = 0
        if self.bc_left == "dirichlet":
            boundary_condition_left = 0
        elif self.bc_left == "neumann":
            boundary_condition_left = 1
        if self.bc_right == "dirichlet":
            boundary_condition_right = 0
        elif self.bc_right == "neumann":
            boundary_condition_right = 1

        if(boundary_condition_left is not boundary_condition_right):
            wt = c ** 2 * self.cheb.solve(v, x, {boundary_condition_left: (0, None), boundary_condition_right: (None, 0)}, m=2)
        else:
            wt = c ** 2 * self.cheb.solve(v, x, {boundary_condition_left: (0, 0)}, m=2)

        # Compute du/dt.
        dudt = np.concatenate([vt, wt])

        return dudt


def cheb_points(N: int) -> np.array:
    """
    Get point grid array of Chebyshev extremal points
    """
    return np.cos(np.arange(0, N) * np.pi / (N-1))

class Cheb:
    """
    Class for pseudospectral reconstruction using Chebyshev basis
    """
    # TODO: implement Cheb class in PyTorch
    def __init__(self):
        self.diffmat = {}
        self.bc_mat = {}
        self.basis = {}
        self.D = {}

    def poly(self, n: np.array, x: np.array) -> np.array:
        """
        Chebyshev polynomials for x \in [-1,1],
        see https://en.wikipedia.org/wiki/Chebyshev_polynomials#Trigonometric_definition
        """
        return np.cos(n * np.arccos(x))

    def chebder(self, N: int, m: int) -> np.array:
        """
        Get Chebyshev derivatives of order m
        Args:
            N (int): number of grid points
            m (int): order of derivative
        Returns:
            np.array: derivatives of m-th order
        """
        diffmat = np.zeros((N - m, N))
        for i in range(N):
            c = np.zeros((N,))
            c[i] = 1
            diffmat[:, i] = np.polynomial.chebyshev.chebder(c, m=m)
        return diffmat

    def get_basis(self, x: np.array) -> np.array:
        """Get polynomial basis matrix
        Args:
            x: num points
        Returns:
            np.array: [N, N] basis matrix where N = len(x)
        """
        N = len(x)
        # Memoization
        if N in self.basis:
            return self.basis[N]

        # Get domain
        L = np.abs(x[0] - x[-1])
        x = cheb_points(N)[:, None]
        n = np.arange(N)[None, :]

        # Compute basis
        self.basis[N] = self.poly(n, x)
        return self.basis[N]

    def solve(self, u: np.array, x: np.array, bcs: list, m: int=1) -> np.array:
        """Differentiation matrix with boundary conditions on [-1, 1]
        f = T @ f_hat
        Args:
            u (np.array): field values
            x (np.array): spatial points
            bcs (list): boundary conditions [left boundary, right boundary]
            m (int): order of derivative
        Returns:
            np.array: [N, N] differentation matrix where N = len(x)
        """
        N = len(x)
        key = hash((N, m) + tuple(bcs.items()))

        # Memoization
        if key not in self.diffmat:
            T = self.get_basis(x)
            L = np.abs(x[0] - x[-1])

            # Cut off boundaries
            t0 = T[:1, :]
            t1 = T[-1:, :]
            T_int = T[1:-1, :]

            # Boundary bordering LHS
            bc_mat = []
            for order, bc in bcs.items():
                if order > 0:
                    # Basis derivative matrix
                    D = self.chebder(N, m=order)
                    D *= (-2 / L) ** order

                    # Differentiate the basis on the boundary
                    t0m = t0[:, :-order] @ D
                    t1m = t1[:, :-order] @ D
                else:
                    t0m = t0
                    t1m = t1

                # Add BCs
                if bc[0] is not None and bc[1] is not None:
                    T_int = np.concatenate([t0m, t1m, T_int], 0)
                    bc_mat = np.concatenate([[bc[0]], [bc[1]], bc_mat], 0)

                else:
                    if bc[0] is not None:
                        T_int = np.concatenate([t0m, T_int], 0)
                        bc_mat = np.concatenate([[bc[0]], bc_mat], 0)
                    if bc[1] is not None:
                        T_int = np.concatenate([t1m, T_int], 0)
                        bc_mat = np.concatenate([[bc[1]], bc_mat], 0)


            # Compute inverse
            Tinv = np.linalg.pinv(T_int)
            diffmat = self.chebder(N, m=m)
            diffmat *= (-2 / L) ** m

            self.diffmat[key] = T[:, :-m] @ diffmat @ Tinv
            self.bc_mat[key] = bc_mat

        # Boundary bordering RHS
        # Cut off boundaries
        u = u[1:-1]
        # Add BCs
        u = np.concatenate([self.bc_mat[key], u], 0)

        return self.diffmat[key] @ u
