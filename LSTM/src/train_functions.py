## PENDIENTE: Comprobar train loop

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import accuracy
from typing import Tuple


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device
) -> Tuple[float, float]:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # define metric lists
    loss_list: list[float] = []
    accuracy_list: list[float] = []

    # Set model to train mode
    model.train()

    for inputs, targets in train_loader:
        
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass of our model -> get the predictions made by our model

        outputs = model(inputs)


        # Compute loss
        loss_value = loss(outputs, targets)
        loss_list.append(loss_value.item())

        # Backward pass
        loss_value.backward()

        # Optimize the parameters
        optimizer.step()

        # Compute accuracy
        acc = accuracy(outputs, targets)
        accuracy_list.append(acc)


    # write on tensorboard

    if writer is not None:
        writer.add_scalar("train/loss", np.mean(loss_list), epoch)

    print_every = 1
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Training Loss: {np.mean(loss_list)}, Training Accuracy: {np.mean(accuracy_list)}")

    return np.mean(loss_list), np.mean(accuracy_list)



@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # define metric lists
    loss_list: list[float] = []
    accuracy_list: list[float] = []

    # Set model to train mode
    model.eval()

    for inputs, targets in val_loader:
        
        
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.float()
        # Forward pass of our model -> get the predictions made by our model
        outputs = model(inputs)

        # Compute loss
        loss_value = loss(outputs, targets)
        loss_list.append(loss_value.item())

        # Compute accuracy
        acc = accuracy(outputs, targets)
        accuracy_list.append(acc)


    # write on tensorboard
    if writer is not None:
        writer.add_scalar("val/loss", np.mean(loss_list), epoch)


    # Print losses
    # Define print cadence
    print_every = 1
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Val Loss: {np.mean(loss_list)}, Val Accuracy: {np.mean(accuracy_list)}")

    return np.mean(loss_list), np.mean(accuracy_list)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    model.eval()
    accuracy_list = []

    with torch.no_grad():

        for inputs, targets in test_loader:


            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)

            acc = accuracy(outputs, targets)
            accuracy_list.append(acc)

    return  np.mean(accuracy_list)
