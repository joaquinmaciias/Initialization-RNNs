# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.utils import set_seed
from src.train_functions import t_step
from src.utils import load_model

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name):
    """
    This function is the main program.
    """
     # hyperparameters
    epochs: int = 70
    lr: float = 5e-4
    batch_size: int = 64
    hidden_dim: int = 128
    num_workers: int = 4
    dropout: float = 0.2
    
    # define initializations
    initializations = ["identity","identity_001","zeros", "constant", "random_normal", "random_uniform",\
                        "truncated_normal", "xavier", "normalized_xavier", "kaiming", "orthogonal"]

    # load data
    _, _, test_data, mean, std = load_data(DATA_PATH, batch_size=batch_size)

    for initialization in initializations:
        print(f"Evaluating model with initialization: {initialization}")

        # define name
        name: str = f"model_rnn_{initialization}"

        # load model
        model: RecursiveScriptModule = load_model(name).to(device)

        # get mae
        t_step(model, test_data, mean, std, device)

if __name__ == "__main__":
    print(f"MAE: {main('best_model')}")
