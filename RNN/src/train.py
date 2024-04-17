# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model
import src.initializations as inits

# other libraries
from tqdm.auto import tqdm
from typing import Final

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # hyperparameters
    epochs: int = 2  #70
    lr: float = 5e-4
    batch_size: int = 64
    hidden_dim: int = 128
    num_workers: int = 4
    dropout: float = 0.2
    
    # define initializations
    initializations = ["identity","identity_001","zeros", "constant", "random_normal", "random_uniform",\
                        "truncated_normal", "xavier", "normalized_xavier", "kaiming", "orthogonal"]
    
     # load data
    train_data, val_data, _, mean, std = load_data(
            DATA_PATH, batch_size=batch_size, num_workers=num_workers)
    
    for initialization in initializations:
        print(f"Training model with initialization: {initialization}")

        # empty nohup file
        open("nohup.out", "w").close()

        # define name and writer
        name: str = f"model_rnn_{initialization}"
        writer: SummaryWriter = SummaryWriter(f"runs/{name}")

        # define model
        model: torch.nn.Module = MyModel(hidden_dim, dropout, initialization).to(device)

        # define loss function
        loss: torch.nn.Module = torch.nn.L1Loss()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in tqdm(range(epochs)):
            train_step(model, train_data, initialization, mean, std, loss, optimizer, writer, epoch, device)
            val_step(model, val_data,initialization, mean, std, loss, None, writer, epoch, device)

        # save model
        save_model(model, name)

    return None


if __name__ == "__main__":
    main()
