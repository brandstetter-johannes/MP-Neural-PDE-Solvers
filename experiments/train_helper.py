import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, GraphCreator
from equations.PDEs import *

def training_loop(model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for (u_base, u_super, x, variables) in loader:
        optimizer.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)
        else:
            data, labels = data.to(device), labels.to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                if f'{model}' == 'GNN':
                    pred = model(graph)
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
                else:
                    data = model(data)
                    labels = labels.to(device)

        if f'{model}' == 'GNN':
            pred = model(graph)
            loss = criterion(pred, graph.y)
        else:
            pred = model(data)
            loss = criterion(pred, labels)

        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()

    losses = torch.stack(losses)
    return losses

def test_timestep_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred = model(data)
                    loss = criterion(pred, labels)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print(f'Step {step}, mean loss {torch.mean(losses)}')



def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    losses_base = []
    for (u_base, u_super, x, variables) in loader:
        losses_tmp = []
        losses_base_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels) / nx_base_resolution

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution
                else:
                    labels = labels.to(device)
                    pred = model(pred)
                    loss = criterion(pred, labels) / nx_base_resolution
                losses_tmp.append(loss / batch_size)

            # Losses for numerical baseline
            for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
                              graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels_super = graph_creator.create_data(u_super, same_steps)
                _, labels_base = graph_creator.create_data(u_base, same_steps)
                loss_base = criterion(labels_super, labels_base) / nx_base_resolution
                losses_base_tmp.append(loss_base / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return losses




