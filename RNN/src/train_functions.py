# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    initialization: str,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        initialization: initialization of the model.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    losses: list[float] = []

    # set model to training mode
    model.train()

    for inputs, targets in train_data:
        # move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # inputs and targets to float
        inputs = inputs.float()
        targets = targets.float()

        outputs = model(inputs)

        # Desnormalization
        outputs = outputs * std + mean
        targets = targets * std + mean

        loss_value = loss(outputs, targets)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    initialization: str,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        initialization: initialization of the model.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    losses: list[float] = []

    # set model to evaluation mode
    model.eval()

    for inputs, targets in val_data:
        # move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # inputs and targets to float
        inputs = inputs.float()
        targets = targets.float()

        outputs = model(inputs)

        # Desnormalization
        outputs = outputs * std + mean
        targets = targets * std + mean

        loss_value = loss(outputs, targets)

        losses.append(loss_value.item())

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
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

    # TODO
    # set model to evaluation mode
    model.eval()

    mae: float = 0.0

    for inputs, targets in test_data:
        # move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # inputs and targets to float
        inputs = inputs.float()
        targets = targets.float()

        outputs = model(inputs)

        # Desnormalization
        outputs = outputs * std + mean
        targets = targets * std + mean

        mae += torch.mean(torch.abs(outputs - targets)).item()

    return mae / len(test_data)
