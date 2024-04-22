import torch
from torch.utils.data import DataLoader
import torchvision


def load_data(batch_size: int, num_workers: int) ->\
        tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Load MNIST data from torchvision.datasets

    Args:
        batch_size: int
        num_workers: int

    Returns:
        train_data: torch.utils.data.DataLoader
        val_data: torch.utils.data.DataLoader
        test_data: torch.utils.data.DataLoader
    '''

    # load data
    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root='./data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root='./data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root='./data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data, val_data, test_data
