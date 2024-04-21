# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final
import os

# own modules
from src.data_MNIST import load_data
from src.utils import set_seed
from src.train_functions import test_step
from src.utils import load_model
from src.plot import accuracy_histogram

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
    epochs: int = 20
    batch_size: int = 128
    hidden_dim: int = 128
    num_workers: int = 4
    input_dim: int = 28
    sequence_length: int = 28

    # define initializations
    initializations = ["zeros", "constant05", "constant_05", "random_normal", "random_uniform",\
                        "truncated_normal", "xavier", "normalized_xavier", "kaiming", "orthogonal"]

    # load data MNIST
    _, _, test_data = load_data(batch_size=batch_size, num_workers=num_workers)

    accuracies: dict = {}
      
    for initialization in initializations:
        print(f"Evaluating model with initialization: {initialization}")

        # define name
        name: str = f"model_rnn_batch_{batch_size}_hidden_{hidden_dim}_init_{initialization}"

        # load model
        model: RecursiveScriptModule = load_model(name).to(device)

        # get mae
        accuracy = test_step(model, test_data, input_dim, sequence_length, device)

        accuracies[initialization] = accuracy

        # print results
        print(f"{initialization} - Accuracy: {accuracy}")

    # plot histogram
    accuracy_histogram(accuracies)

if __name__ == "__main__":
    main("model_rnn")
