import os
import math
import numpy as np
import torch
import common.coefficients as coefficients
from torch import nn
from scipy.ndimage import gaussian_filter
from scipy.special import poch
from torch.nn import functional as F

class FDM():
    """
    FDM reconstruction
    """
    def __init__(self, pde, device: torch.cuda.device="cpu") -> None:
        """
        Initialize FDM reconstruction class
        Args:
            pde (PDE): PDE at hand
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.pde = pde
        self.weights1 = torch.tensor(coefficients.FDM_derivatives[1]).to(self.device)
        self.weights2 = torch.tensor(coefficients.FDM_derivatives[2]).to(self.device)
        self.weights3 = torch.tensor(coefficients.FDM_derivatives[3]).to(self.device)
        self.weights4 = torch.tensor(coefficients.FDM_derivatives[4]).to(self.device)

    def pad(self, input: torch.Tensor) -> torch.Tensor:
        """
        Padding according to FDM derivatives for periodic boundary conditions
        Padding with size 2 is correct for 4th order accuracy for first and second derivative and
        for 2nd order accuracy for third and fourth derivative (for simplicity)
        """
        left = input[..., -3:-1]
        right = input[..., 1:3]
        padded_input = torch.cat([left, input, right], -1)
        return padded_input

    def first_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
        FDM method for first order derivative
        """
        return (1 / self.pde.dx) * F.conv1d(input, self.weights1)

    def second_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for second order derivative
        """
        return (1 / self.pde.dx)**2 * F.conv1d(input, self.weights2)

    def third_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for third order derivative
        """
        return (1 / self.pde.dx)**3 * F.conv1d(input, self.weights3)

    def fourth_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for fourth order derivative
        """
        return (1 / self.pde.dx)**4 * F.conv1d(input, self.weights4)


class WENO():
    """
    WENO5 reconstruction
    """
    def __init__(self, pde, order: int=3, device: torch.cuda.device="cpu") -> None:
        """
        Initialization of GPU compatible WENO5 method
        Args:
            pde (PDE): PDE at hand
            order (int): order of WENO coefficients (order 3 for WENO5 method)
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()

        self.pde = pde
        self.order = order
        self.epsilon = 1e-16
        self.device = device

        assert(self.order == 3) # as default for WENO5 scheme, higher orders are not implemented
        betaA = coefficients.betaA_all[self.order]
        betaB = coefficients.betaB_all[self.order]
        gamma = coefficients.gamma_all[self.order]
        stencils = coefficients.stencils_all[self.order]

        self.betaAm = torch.tensor(betaA).to(self.device)
        self.betaBm = torch.tensor(betaB).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.stencils = torch.tensor(stencils).to(self.device)

    def pad(self, input: torch.Tensor) -> torch.Tensor:
        """
        Padding according to order of Weno scheme
        """
        left = input[..., -self.order:-1]
        right = input[..., 1:self.order]
        padded_input = torch.cat([left, input, right], -1)
        return padded_input

    def reconstruct_godunov(self, input: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct via Godunov flux
        Args:
            input (torch.Tensor): padded input
            dx (torch.Tensor): step size
        Returns:
            torch.Tensor: reconstructed Godunov flux
        """
        # reconstruct from the right
        rec_plus = self.reconstruct(torch.flip(input, [-1]))
        rec_plus = torch.flip(rec_plus, [-1])
        rec_plus = torch.roll(rec_plus, -1, -1)
        # reconstruct from the left
        rec_minus = self.reconstruct(input)

        switch = torch.ge(rec_plus, rec_minus).type(torch.float64)
        flux_plus = self.pde.flux(rec_plus)
        flux_minus = self.pde.flux(rec_minus)
        min_flux = torch.min(flux_minus, flux_plus)
        max_flux = torch.max(flux_minus, flux_plus)
        flux_out = switch * min_flux + (1 - switch) * max_flux
        flux_in = torch.roll(flux_out, 1, -1)
        flux_godunov = 1 / dx * (flux_out - flux_in)
        return flux_godunov


    def reconstruct_laxfriedrichs(self, input: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct via Lax-Friedrichs flux
        Args:
            input (torch.Tensor): padded input
            dx (torch.Tensor): step size
        Returns:
            torch.Tensor: reconstructed Lax-Friedrichs flux
        """
        f = self.pde.flux(input)
        alpha = torch.max(input, -1).values
        f_plus = f + alpha * input
        f_minus = f-alpha * input

        # construct flux from the left
        flux_plus = self.reconstruct(f_plus) / 2
        # construct flux from the right
        flux_minus = self.reconstruct(torch.flip(f_minus, [-1])) / 2
        flux_minus = torch.flip(flux_minus, [-1])
        flux_minus = torch.roll(flux_minus, -1, -1)
        # add fluxes
        flux_out = flux_plus + flux_minus
        flux_in = torch.roll(flux_out, 1, -1)
        flux_laxfriedrichs = 1 / dx * (flux_out - flux_in)
        return flux_laxfriedrichs


    def reconstruct(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Weno5 reconstruction
        '''

        b1 = F.conv1d(input, self.betaAm)
        b2 = F.conv1d(input, self.betaBm)
        beta = b1 * b1 + b2 * b2

        w_tilde = self.gamma / (self.epsilon + beta) ** 2
        w = (w_tilde / torch.sum(w_tilde, axis=1, keepdim=True)).view(-1, 1, 3, w_tilde.shape[-1])

        derivatives = F.conv1d(input, self.stencils).view(input.shape[0], -1, 3, w.shape[-1])
        output = torch.sum(w * derivatives, axis=2)

        return output

