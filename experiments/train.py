import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, GraphCreator
from experiments.models_gnn import MP_PDE_Solver
from experiments.models_cnn import BaseCNN
from experiments.train_helper import *
from equations.PDEs import *

def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'experiments/log'):
        os.mkdir(f'experiments/log')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')

def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()

    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    for i in range(graph_creator.t_res):
        losses = training_loop(model, unrolling, args.batch_size, optimizer, loader, graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}')

def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    # next we test the unrolled losses
    losses = test_unrolled_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  nr_gt_steps=args.nr_gt_steps,
                                  nx_base_resolution=args.base_resolution[1],
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    return torch.mean(losses)


def main(args: argparse):

    device = args.device
    check_directory()

    base_resolution = args.base_resolution
    super_resolution = args.super_resolution

    # Check for experiments and if resolution is available
    if args.experiment == 'E1' or args.experiment == 'E2' or args.experiment == 'E3':
        pde = CE(device=device)
        assert(base_resolution[0] == 250)
        assert(base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40)
    elif args.experiment == 'WE1' or args.experiment == 'WE2' or args.experiment == 'WE3':
        pde = WE(device=device)
        assert (base_resolution[0] == 250)
        assert (base_resolution[1] == 100 or base_resolution[1] == 50 or base_resolution[1] == 40 or base_resolution[1] == 20)
        if args.model != 'GNN':
            raise Exception("Only MP-PDE Solver is implemented for irregular grids so far.")
    else:
        raise Exception("Wrong experiment")

    # Load datasets
    train_string = f'data/{pde}_train_{args.experiment}.h5'
    valid_string = f'data/{pde}_valid_{args.experiment}.h5'
    test_string = f'data/{pde}_test_{args.experiment}.h5'
    try:
        train_dataset = HDF5Dataset(train_string, pde=pde, mode='train', base_resolution=base_resolution, super_resolution=super_resolution)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

        valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', base_resolution=base_resolution, super_resolution=super_resolution)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

        test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', base_resolution=base_resolution, super_resolution=super_resolution)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)
    except:
        raise Exception("Datasets could not be loaded properly")

    # Equation specific parameters
    pde.tmin = train_dataset.tmin
    pde.tmax = train_dataset.tmax
    pde.grid_size = base_resolution

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

    if(args.log):
        logfile = f'experiments/log/{args.model}_{pde}_{args.experiment}_xresolution{args.base_resolution[1]}-{args.super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    save_path = f'models/GNN_{pde}_{args.experiment}_xresolution{args.base_resolution[1]}-{args.super_resolution[1]}_n{args.neighbors}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.pt'
    print(f'Training on dataset {train_string}')
    print(device)
    print(save_path)

    # Equation specific input variables
    eq_variables = {}
    if not args.parameter_ablation:
        if args.experiment == 'E2':
            print(f'Beta parameter added to the GNN solver')
            eq_variables['beta'] = 0.2
        elif args.experiment == 'E3':
            print(f'Alpha, beta, and gamma parameter added to the GNN solver')
            eq_variables['alpha'] = 3.
            eq_variables['beta'] = 0.4
            eq_variables['gamma'] = 1.
        elif (args.experiment == 'WE3'):
            print('Boundary parameters added to the GNN solver')
            eq_variables['bc_left'] = 1
            eq_variables['bc_right'] = 1

    graph_creator = GraphCreator(pde=pde,
                                 neighbors=args.neighbors,
                                 time_window=args.time_window,
                                 t_resolution=args.base_resolution[0],
                                 x_resolution=args.base_resolution[1]).to(device)

    if args.model == 'GNN':
        model = MP_PDE_Solver(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables).to(device)
    elif args.model == 'BaseCNN':
        model = BaseCNN(pde=pde,
                        time_window=args.time_window).to(device)
    else:
        raise Exception("Wrong model specified")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_decay)

    # Training loop
    min_val_loss = 10e30
    test_loss = 10e30
    criterion = torch.nn.MSELoss(reduction="sum")
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train(args, pde, epoch, model, optimizer, train_loader, graph_creator, criterion, device=device)
        print("Evaluation on validation dataset:")
        val_loss = test(args, pde, model, valid_loader, graph_creator, criterion, device=device)
        if(val_loss < min_val_loss):
            print("Evaluation on test dataset:")
            test_loss = test(args, pde, model, test_loader, graph_creator, criterion, device=device)
            # Save model
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}\n")
            min_val_loss = val_loss

        scheduler.step()

    print(f"Test loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')
    parser.add_argument('--parameter_ablation', type=eval, default=False,
                        help='Flag for ablating MP-PDE solver without equation specific parameters')

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 100], help="PDE base resolution on which network is applied")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")
    parser.add_argument('--neighbors', type=int,
                        default=3, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--time_window', type=int,
                        default=25, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=1, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--nr_gt_steps', type=int,
                        default=2, help="Number of steps done by numerical solver")

    # Misc
    parser.add_argument('--print_interval', type=int, default=20,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')

    args = parser.parse_args()
    main(args)
