import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import h5py
import random
from scipy.integrate import solve_ivp
from copy import copy
from typing import Callable, Tuple
from datetime import datetime
from equations.PDEs import PDE, CE, WE
from temporal.solvers import *


def check_files(pde: dict, modes: dict, experiment: str) -> None:
    """
    Check if data files exist and replace them if wanted.
    Args:
        pde (dict): dictionary of PDEs at different resolutions
        modes (dict): mode ([train, valid, test]), replace, num_samples, training suffix
        experiment (str): experiment string
    Returns:
            None
    """
    for mode, replace, num_samples in modes:
        save_name = "data/" + "_".join([str(pde[list(pde.keys())[0]]), mode]) + "_" + experiment
        if (replace == True):
            if os.path.exists(f'{save_name}.h5'):
                os.remove(f'{save_name}.h5')
                print(f'File {save_name}.h5 is deleted.')
            else:
                print(f'No file {save_name}.h5 exists yet.')
        else:
            print(f'File {save_name}.h5 is kept.')


def check_directory() -> None:
    """
    Check if data and log directories exist, and create otherwise
    Args:
    Returns:
        None
    """
    if os.path.exists(f'data'):
        print(f'Data directory exists and will be written to.')
    else:
        os.mkdir(f'data')
        print(f'Data directory created.')
    if not os.path.exists(f'data/log'):
        os.mkdir(f'data/log')


def cheb_grid(xmin: float, xmax: float, N: int) -> list:
    """
    Get Chebyshev grid depending on xmin, xmax and number of grid points
    Args:
         xmin (float): minimum x value
         xmax (float): maximum x value
         N (int): number of points
    Returns:
        list: list of grid points
    """
    x = np.cos(np.arange(0, N) * np.pi / (N - 1))
    x = x[::-1]
    # Shift [-1, +1] to [xmin, xmax]
    xnorm = (x + 1.) / 2.
    return (xmax - xmin) * xnorm + xmin


def initial_conditions(A: torch.Tensor,
                       omega: torch.Tensor,
                       phi: torch.Tensor,
                       l: torch.Tensor,
                       pde: PDE) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Return initial conditions for combined equation based on initial parameters
    Args:
        A (torch.Tensor): amplitude of different sine waves
        omega (torch.Tensor): time-dependent frequency
        phi (torch.Tensor): phase shift of different sine waves
        l (torch.Tensor): frequency of sine waves
    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function which initializes for chosen set of parameters
    """
    def fnc(x, t=0):
        u = torch.sum(A * torch.sin(omega*t + (2 * np.pi * l * x / pde.L) + phi), -1)
        return u
    return fnc


def params(pde: PDE,
           batch_size: int,
           device: torch.cuda.device="cpu",) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get initial parameters for combined equation
    Args:
        pde (PDE): PDE at hand
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A, omega, phi, l
    """
    A = torch.rand(batch_size, 1, pde.N) - 0.5
    omega = 0.8 * (torch.rand(batch_size, 1, pde.N) - 0.5)
    phi = 2.0 * np.pi * torch.rand(batch_size, 1, pde.N)
    l = torch.randint(pde.lmin, pde.lmax, (batch_size, 1, pde.N))
    return A.to(device), omega.to(device), phi.to(device), l.to(device)


def generate_data_wave_equation(experiment: str,
                                boundary_condition: str,
                                pde: dict,
                                mode: str,
                                num_samples: int=1,
                                batch_size: int=1,
                                wave_speed: float=2.,
                                device: torch.cuda.device="cpu") -> None:
    """
    Generate data for wave equation using different boundary conditions
    Args:
        experiment (str): experiment string
        boundary_condition (str): boundary condition string
        pde (dict): dictionary for PDEs at different resolution
        mode (str): train, valid, test
        num_samples (int): number of trajectories to solve
        batch_size (int): batch size
        wave_speed (float): how fast the wave is moving
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Device: {device}')
    num_batches = num_samples // batch_size

    pde_string = str(pde[list(pde.keys())[0]])
    print(f'Equation: {pde_string}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')
    print(f'Batch size: {batch_size}')
    print(f'Number of batches: {num_batches}')

    sys.stdout.flush()

    save_name = "data/" + "_".join([str(pde[list(pde.keys())[0]]), mode]) + "_" + experiment
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    t = {}
    x = {}
    h5f_u = {}
    h5f_bc_right = {}
    h5f_bc_left = {}
    h5f_c = {}
    h5f_valid = {}

    tol = 1e-3

    # TODO: implement data generation in PyTorch
    for key in pde:
        t[key] = np.linspace(pde[key].tmin, pde[key].tmax, pde[key].grid_size[0])
        x[key] = cheb_grid(pde[key].xmin, pde[key].xmax, pde[key].nx)
        h5f_u[key] = dataset.create_dataset(key, (num_samples, pde[key].grid_size[0], pde[key].grid_size[1]), dtype=float)
        h5f[mode][key].attrs['dt'] = pde[key].dt
        h5f[mode][key].attrs['dx'] = pde[key].dx
        h5f[mode][key].attrs['nt'] = pde[key].grid_size[0]
        h5f[mode][key].attrs['nx'] = pde[key].grid_size[1]
        h5f[mode][key].attrs['tmin'] = pde[key].tmin
        h5f[mode][key].attrs['tmax'] = pde[key].tmax
        h5f[mode][key].attrs['x'] = x[key]

    h5f_bc_left['bc_left'] = dataset.create_dataset('bc_left', (num_samples, ), dtype=int)
    h5f_bc_right['bc_right'] = dataset.create_dataset('bc_right', (num_samples, ), dtype=int)
    h5f_c['c'] = dataset.create_dataset('c', (num_samples, ), dtype=float)

    # For now it only works for batch_size==1
    # TODO: implement parallelized data generation
    if batch_size > 1:
        raise NotImplementedError

    for idx in range(num_batches):

        # Initial condition for Gaussian wave
        start = np.random.uniform(-4., 4., 1)
        c = wave_speed
        bc_left, bc_right = 0, 0
        if boundary_condition == "dirichlet":
            bc_left = 0
            bc_right = 0
        elif boundary_condition == "neumann":
            bc_left = 1
            bc_right = 1
        elif boundary_condition == "mixed":
            bc_left = np.random.randint(0, 2, size=1)
            bc_right = np.random.randint(0, 2, size=1)
        else:
            raise Exception("Wrong boundary conditions")

        skip = 0
        sol = {}
        for key in pde:
            if bc_left == 0:
                pde[key].bc_left = "dirichlet"
            else:
                pde[key].bc_left = "neumann"
            if bc_right == 0:
                pde[key].bc_right = "dirichlet"
            else:
                pde[key].bc_right = "neumann"

            # Get initial Gaussian blob
            u = np.exp(-(x[key] - start) ** 2)
            v = -2 * c * (x[key] - start) * u
            u0 = np.concatenate([u, v])

            # Solving for the full trajectories and runtime measurement using pseudospectral Radau method
            torch.cuda.synchronize()
            t1 = time.time()
            # Spatial derivatives are calculated using chebdx method
            solved = solve_ivp(pde[key].chebdx, [t[key][0], t[key][-1]], u0, method='Radau', t_eval=t[key], args=(x[key], c), rtol=tol, atol=tol)
            torch.cuda.synchronize()
            t2 = time.time()
            print(f'{key}: {t2 - t1:.4f}s')

            y = solved.y.T[::-1]
            y = y[:, :y.shape[-1] // 2]
            sol[key] = y

        # save
        for key in pde:
            h5f_u[key][idx:idx+1, :, :] = sol[key]

        h5f_bc_left['bc_left'][idx:idx+1] = bc_left
        h5f_bc_right['bc_right'][idx:idx+1] = bc_right
        h5f_c['c'][idx:idx+1] = c

        print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))

        sys.stdout.flush()

    print()

    print("Data saved")
    print()
    print()
    h5f.close()


def generate_data_combined_equation(experiment: str,
                                    pde: dict,
                                    mode: str,
                                    num_samples: int = 1,
                                    batch_size: int = 1,
                                    device: torch.cuda.device = "cpu",
                                    alpha: list = [1., 1.],
                                    beta: list = [0., 0.],
                                    gamma: list = [0., 0.]) -> None:

    """
    Generate data for combined equation using different coefficients
    Args:
        experiment (str): experiment string
        pde (dict): dictionary for PDEs at different resolution
        mode (str): train, valid, test
        num_samples (int): number of trajectories to solve
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
        alpha (list): alpha parameter (low, high)
        beta (list): beta parameter (low, high)
        gamma (list): gamma parameter (low, high)
    Returns:
        None
    """
    print(f'Device: {device}')
    num_batches = num_samples // batch_size

    pde_string = str(pde[list(pde.keys())[0]])
    print(f'Equation: {experiment}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')
    print(f'Batch size: {batch_size}')
    print(f'Number of batches: {num_batches}')

    save_name = "data/" + "_".join([str(pde[list(pde.keys())[0]]), mode]) + "_" + experiment
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    t = {}
    x = {}
    h5f_u = {}
    h5f_alpha = {}
    h5f_beta = {}
    h5f_gamma = {}

    for key in pde:
        t[key] = torch.linspace(pde[key].tmin, pde[key].tmax, pde[key].grid_size[0]).to(device)
        x[key] = torch.linspace(0, pde[key].L, pde[key].grid_size[1]).to(device)
        h5f_u[key] = dataset.create_dataset(key, (num_samples, pde[key].grid_size[0], pde[key].grid_size[1]), dtype=float)
        h5f[mode][key].attrs['dt'] = pde[key].dt
        h5f[mode][key].attrs['dx'] = pde[key].dx
        h5f[mode][key].attrs['nt'] = pde[key].grid_size[0]
        h5f[mode][key].attrs['nx'] = pde[key].grid_size[1]
        h5f[mode][key].attrs['tmin'] = pde[key].tmin
        h5f[mode][key].attrs['tmax'] = pde[key].tmax
        h5f[mode][key].attrs['x'] = x[key].cpu()

    h5f_alpha['alpha'] = dataset.create_dataset('alpha', (num_samples, ), dtype=float)
    h5f_beta['beta'] = dataset.create_dataset('beta', (num_samples, ), dtype=float)
    h5f_beta['gamma'] = dataset.create_dataset('gamma', (num_samples, ), dtype=float)

    # torch.random.manual_seed(2)
    for idx in range(num_batches):

        A, omega, phi, l = params(pde[list(pde.keys())[0]], batch_size, device=device)

        # Time dependent force term
        def force(t):
            return initial_conditions(A, omega, phi, l, pde[key])(x[key][:, None], t)[:, None]

        if alpha[0] is not alpha[1]:
            alpha_ = torch.distributions.Uniform(low=alpha[0], high=alpha[1]).sample((1, 1)).to(device)
        else:
            alpha_ = torch.tensor([[alpha[0]]]).to(device)
        if beta[0] is not beta[1]:
            beta_ = torch.distributions.Uniform(low=beta[0], high=beta[1]).sample((1, 1)).to(device)
        else:
            beta_ = torch.tensor([[beta[0]]]).to(device)
        if gamma[0] is not gamma[1]:
            gamma_ = torch.distributions.Uniform(low=gamma[0], high=gamma[1]).sample((1, 1)).to(device)
        else:
            gamma_ = torch.tensor([[gamma[0]]]).to(device)

        sol = {}
        for key in pde:
            # Initialize PDE parameters and get initial condition
            pde[key].alpha = alpha_
            pde[key].beta = beta_
            pde[key].gamma = gamma_
            pde[key].force = force
            u0 = initial_conditions(A, omega, phi, l, pde[key])(x[key][:, None])
            # The spatial method is the WENO reconstruction for uux and FD for the rest
            spatial_method = pde[key].WENO_reconstruction

            # Solving full trajectories and runtime measurement
            torch.cuda.synchronize()
            t1 = time.time()
            solver = Solver(RKSolver(Dopri45(), device=device), spatial_method)
            sol[key] = solver.solve(x0=u0[:, None].to(device), times=t[key][None, :].to(device))
            torch.cuda.synchronize()
            t2 = time.time()
            print(f'{key}: {t2 - t1:.4f}s')

        # Save solutions
        for key in pde:
            h5f_u[key][idx * batch_size:(idx + 1) * batch_size, :, :] = \
                sol[key].cpu().reshape(batch_size, pde[key].grid_size[0], -1)

        h5f_alpha['alpha'][idx * batch_size:(idx + 1) * batch_size] = alpha_.detach().cpu()
        h5f_beta['beta'][idx * batch_size:(idx + 1) * batch_size] = beta_.detach().cpu()
        h5f_beta['gamma'][idx * batch_size:(idx + 1) * batch_size] = gamma_.detach().cpu()

        print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))

        sys.stdout.flush()

    print()

    print("Data saved")
    print()
    print()
    h5f.close()


def combined_equation(experiment: str,
                      starting_time: float = 0.0,
                      end_time: float = 4.0,
                      num_samples_train: int = 2 ** 5,
                      num_samples_valid: int = 2 ** 5,
                      num_samples_test: int = 2 ** 5,
                      batch_size: int = 4,
                      device: torch.cuda.device="cpu",
                      alpha: list = [1., 1.],
                      beta: list = [0., 0.],
                      gamma: list = [0., 0]) -> None:
    """
    Setting up method, files and PDEs for combined equation with alpha, beta, gamma parameters
    Args:
        experiment (str): experiment string
        starting_time (float): start of trajectory
        end_time (float): end of trajectory
        num_samples_train (int): training samples
        num_samples_valid (int): validation samples
        num_samples_test (int): test samples
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
        alpha (list): alpha parameter (low, high)
        beta (list): beta parameter (low, high)
        gamma (list): gamma parameter (low, high)
    Returns:
        None
    """
    # Different temporal and spatial resolutions
    nt = (250, 250, 250, 250)
    nx = (200, 100, 50, 40)
    nt_max = max(nt)
    nx_max = max(nx)

    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    if args.log:
        logfile = f'data/log/CE_{experiment}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    # Create instances of PDE for each (nt, nx)
    pde = {}
    for nt_, nx_ in zip(nt, nx):
        pde[f'pde_{nt_}-{nx_}'] = CE(starting_time, end_time, (nt_, nx_), device=device)

    # Check if train, valid and test files already exist and replace if wanted
    replace = True
    files = {("train", replace, num_samples_train),
             ("valid", replace, num_samples_valid),
             ("test", replace, num_samples_test)}
    check_files(pde, files, experiment=experiment)

    for mode, _, num_samples in files:
        generate_data_combined_equation(experiment=experiment,
                                        pde=pde,
                                        mode=mode,
                                        num_samples=num_samples,
                                        batch_size=batch_size,
                                        device=device,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma)


def wave_equation(experiment: str,
                  boundary_condition: str,
                  starting_time: float = 0.0,
                  end_time: float = 100.0,
                  num_samples_train: int = 2 ** 5,
                  num_samples_valid: int = 2 ** 5,
                  num_samples_test: int = 2 ** 5,
                  wave_speed: float = 2.,
                  batch_size: int = 1,
                  device: torch.cuda.device="cpu") -> None:
    """
    Setting up method, files and PDEs for combined equation with alpha, beta, gamma parameters
    Args:
        experiment (str): experiment string
        boundary_condition (str): boundary condition string
        starting_time (float): start of trajectory
        end_time (float): end of trajectory
        num_samples_train (int): training samples
        num_samples_valid (int): validation samples
        num_samples_test (int): test samples
        wave_speed (float): speed at which wave travels
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    # Parallel data production for batch_size > 1 is not implemented yet
    if batch_size > 1:
        raise NotImplementedError

    # Different temporal and spatial resolutions
    nt = (250, 250, 250, 250, 250)
    nx = (200, 100, 50, 40, 20)
    nt_max = max(nt)
    nx_max = max(nx)

    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    if args.log:
        logfile = f'data/log/WE_{experiment}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    # Create instances of PDE for each resolution (nt, nx)
    replace = True
    pde = {}
    for nt_, nx_ in zip(nt, nx):
        pde[f'pde_{nt_}-{nx_}'] = WE(tmin=starting_time, tmax=end_time, grid_size=(nt_, nx_), device=device)

    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", replace, num_samples_train),
             ("valid", replace, num_samples_valid),
             ("test", replace, num_samples_test)}
    check_files(pde, files, experiment=experiment)

    for mode, _, num_samples in files:
        generate_data_wave_equation(experiment=experiment,
                                    boundary_condition=boundary_condition,
                                    pde=pde,
                                    mode=mode,
                                    num_samples=num_samples,
                                    wave_speed=wave_speed,
                                    batch_size=batch_size,
                                    device=device)


def main(args):
    """
        Main method for data generation.
    """
    check_directory()

    if args.experiment == 'E1':
        combined_equation(experiment=args.experiment,
                          starting_time=0.0,
                          end_time=4.0,
                          num_samples_train=args.train_samples,
                          num_samples_valid=args.valid_samples,
                          num_samples_test=args.test_samples,
                          batch_size=args.batch_size,
                          device=args.device,
                          alpha=[1., 1.],
                          beta=[0., 0.],
                          gamma=[0., 0.])

    elif args.experiment == 'E2':
        combined_equation(experiment=args.experiment,
                          starting_time=0.0,
                          end_time=4.0,
                          num_samples_train=args.train_samples,
                          num_samples_valid=args.valid_samples,
                          num_samples_test=args.test_samples,
                          batch_size=args.batch_size,
                          device=args.device,
                          alpha=[1., 1.],
                          beta=[0., 0.2],
                          gamma=[0., 0.])

    elif args.experiment == 'E3':
        combined_equation(experiment=args.experiment,
                          starting_time=0.0,
                          end_time=2.0,
                          num_samples_train=args.train_samples,
                          num_samples_valid=args.valid_samples,
                          num_samples_test=args.test_samples,
                          batch_size=args.batch_size,
                          device=args.device,
                          alpha=[0., 6.],
                          beta=[0.1, 0.4],
                          gamma=[0., 1.])

    elif args.experiment == "WE1":
        wave_equation(experiment=args.experiment,
                      boundary_condition="dirichlet",
                      starting_time=0.0,
                      end_time=100.0,
                      num_samples_train=args.train_samples,
                      num_samples_valid=args.valid_samples,
                      num_samples_test=args.test_samples,
                      wave_speed=args.wave_speed,
                      batch_size=1,
                      device=args.device)

    elif args.experiment == "WE2":
        wave_equation(experiment=args.experiment,
                      boundary_condition="neumann",
                      starting_time=0.0,
                      end_time=100.0,
                      num_samples_train=args.train_samples,
                      num_samples_valid=args.valid_samples,
                      num_samples_test=args.test_samples,
                      wave_speed=args.wave_speed,
                      batch_size=1,
                      device=args.device)

    elif args.experiment == "WE3":
        wave_equation(experiment=args.experiment,
                      boundary_condition="mixed",
                      starting_time=0.0,
                      end_time=100.0,
                      num_samples_train=args.train_samples,
                      num_samples_valid=args.valid_samples,
                      num_samples_test=args.test_samples,
                      wave_speed=args.wave_speed,
                      batch_size=1,
                      device=args.device)

    else:
        raise Exception("Wrong experiment")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for which data should create for: [E1, E2, E3, WE1, WE2, WE3]')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--train_samples', type=int, default=2 ** 5,
                        help='Samples in the training dataset')
    parser.add_argument('--valid_samples', type=int, default=2 ** 5,
                        help='Samples in the validation dataset')
    parser.add_argument('--test_samples', type=int, default=2 ** 5,
                        help='Samples in the test dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size used for creating training, val, and test dataset')
    parser.add_argument('--wave_speed', type=float, default=2.,
                        help='Wave speed, only meaningful if set for wave equation experiments (WE1, WE2, WE3)')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')

    args = parser.parse_args()
    main(args)