# Deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# Other libraries
from typing import Final

# Own modules

from data_processing import load_data
from train_functions import t_step
from src.train import SEED

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(name) -> float:
    """
    This function is the main program.
    """
    # Load test data
    test_loader: DataLoader
    _, _, test_loader = load_data(DATA_PATH, num_workers=4, drop_last=True)

    name = f"models/{name}.pth"
    # Load model
    model = torch.load(name)

    # Set model to evaluation mode
    model.eval()

    # Evaluate model with tstep and get mean MAE score
    accuracy = t_step(model, test_loader, device)

    return accuracy


# if __name__ == "__main__":

#     mae = main("best_model")
#     print(f"MAE Score: {mae}")