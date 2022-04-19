import torch
from collections import OrderedDict
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from equations.PDEs import PDE, CE, WE


class BaseCNN(nn.Module):
    '''
    A simple baseline 1D Res CNN approach, the time dimension is stacked in the channels
    '''
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_channels: int = 40,
                 padding_mode: str = f'circular') -> None:
        """
        Initialize the simple CNN architecture. It contains 8 1D CNN-layers with skip connections
        and increasing receptive field.
        The input to the forward pass has the shape [batch, time_window, x].
        The output has the shape [batch, time_window, x].
        Args:
            pde (PDE): the PDE at hand
            time_window (int): input/output timesteps of the trajectory
            hidden_channels: hidden channel dimension
            padding_mode (str): circular mode as default for periodic boundary problems
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.time_window = time_window
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode

        self.conv1 = nn.Conv1d(in_channels=self.time_window, out_channels=self.hidden_channels, kernel_size=3, padding=1,
                               padding_mode=self.padding_mode, bias=True)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv3 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv4 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2,
                               padding_mode=self.padding_mode, bias=True)
        self.conv5 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv6 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv7 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=7, padding=3,
                               padding_mode=self.padding_mode, bias=True)
        self.conv8 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.time_window, kernel_size=9, padding=4,
                               padding_mode=self.padding_mode, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)



    def __repr__(self):
        return f'BaseCNN'

    def forward(self, u):
        """Forward pass of solver
        """

        x = F.elu(self.conv1(u))
        x = x + F.elu(self.conv2(x))
        x = x + F.elu(self.conv3(x))
        x = x + F.elu(self.conv4(x))
        x = x + F.elu(self.conv5(x))
        x = x + F.elu(self.conv6(x))
        x = x + F.elu(self.conv7(x))
        x = self.conv8(x)

        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(x.device)
        dt = torch.cumsum(dt, dim=1)[None, :, :, None]
        out = u[:, -1, :][:, None, None, :].repeat(1, 1, self.time_window, 1) + dt * x[:, None, :, :]
        return out.squeeze()