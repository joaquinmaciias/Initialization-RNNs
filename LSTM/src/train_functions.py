## PENDIENTE: Comprobar train loop

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



def MAE(outputs, targets):
    return torch.mean(torch.abs(outputs - targets))


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device
) -> None:
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
    mae_list: list[float] = []

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

        # Compute mae
        # mae_val = MAE(outputs, targets)
        # mae_list.append(mae_val.item())

    # write on tensorboard

    if writer is not None:
        writer.add_scalar("train/loss", np.mean(loss_list), epoch)
        #writer.add_scalar("train/mae", np.mean(mae_list), epoch)
    # print('Train mae:', np.mean(mae_list))

    print_every = 1
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Training Loss: {np.mean(loss_list)}")
        #print(f"Epoch {epoch}, Training MAE: {np.mean(mae_list)}")

    return np.mean(loss_list)



@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # define metric lists
    loss_list: list[float] = []
    mae_list: list[float] = []

    # Set model to train mode
    model.eval()

    for inputs, targets in val_loader:
        
        
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass of our model -> get the predictions made by our model
        outputs = model(inputs)

        # Compute loss
        loss_value = loss(outputs, targets)
        loss_list.append(loss_value.item())

        # Compute mae
        # mae_val = MAE(outputs, targets)
        # mae_list.append(mae_val.item())

    # write on tensorboard
    if writer is not None:
        writer.add_scalar("val/loss", np.mean(loss_list), epoch)
        # writer.add_scalar("val/mae", np.mean(mae_list), epoch)
    # print('Validation mae:', np.mean(mae_list))

    # Print losses
    # Define print cadence
    print_every = 1
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Val Loss: {np.mean(loss_list)}")
        #print(f"Epoch {epoch}, Val MAE: {np.mean(mae_list)}")

    return np.mean(loss_list)


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
    test_loss = 0

    with torch.no_grad():

        for inputs, targets in test_loader:
        
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss(outputs, targets)

            test_loss += loss.item()

    return test_loss / len(test_loader)
