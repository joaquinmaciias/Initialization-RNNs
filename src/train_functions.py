# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    input_dim: int,
    sequence_length: int,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> tuple[float, int]:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        initialization: initialization of the model.
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

    for i, (images, labels) in enumerate(train_data):

        # reshape the images
        images = images.view(-1, sequence_length, input_dim)

        # move data to device
        inputs, targets = images.to(device), labels.to(device)

        outputs = model(inputs)

        loss_value = loss(outputs, targets)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        
    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)

    # return the loss values
    return np.mean(losses), epoch


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    input_dim: int,
    sequence_length: int,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> tuple[float, int]:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        initialization: initialization of the model.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    losses: list[float] = []

    # set model to evaluation mode
    model.eval()

    for i, (images, labels) in enumerate(val_data):

        # reshape the images
        images = images.view(-1, sequence_length, input_dim)

        # move data to device
        inputs, targets = images.to(device), labels.to(device)

        outputs = model(inputs)

        loss_value = loss(outputs, targets)

        losses.append(loss_value.item())

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)

    # return the loss values
    return np.mean(losses), epoch


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    input_dim: int,
    sequence_length: int,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    # TODO
    # set model to evaluation mode
    model.eval()

    total : int = 0
    correct : int = 0

    with torch.no_grad():

        for inputs, targets in test_data:

            # reshape the images
            inputs = inputs.view(-1, sequence_length, input_dim)

            # move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # models outputs
            outputs = model(inputs)

            total += targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()

    accuracy: float = 100 * correct / total

    return accuracy
