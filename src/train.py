# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.plot import save_heatmap, plot_loss
from src.data_MNIST import load_data
from src.models import RNN
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model

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
    epochs: int = 2
    lr: float = 0.01
    batch_size: int = 128
    hidden_dim: int = 128
    num_workers: int = 4
    num_layers: int = 2
    input_dim: int = 28
    sequence_length: int = 28
    num_classes: int = 10   # MNIST has 10 classes

    # define initializations - constant05: 0.5, constant_05: -0.5
    initializations = [
        "zeros", "constant05", "constant_05", "random_normal",
        "random_uniform", "truncated_normal", "xavier",
        "normalized_xavier", "kaiming", "orthogonal"
    ]

    # load data MNIST
    train_data, val_data, _ = load_data(batch_size=batch_size, num_workers=num_workers)

    values: dict = {}

    for initialization in initializations:
        print(f"Training model with initialization: {initialization}")

        train_values: dict = {}
        val_values: dict = {}

        # empty nohup file
        open("nohup.out", "w").close()

        # define name and writer
        name: str = (
            f"model_rnn_batch_{batch_size}_hidden_{hidden_dim}_init_{initialization}"
        )
        writer: SummaryWriter = SummaryWriter(f"runs/{name}")

        # define model
        model: torch.nn.Module = RNN(
            input_dim, hidden_dim, num_layers, num_classes, initialization
            ).to(device)
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # save the weights
        save_heatmap(model, 'weights', initialization)

        for epoch in tqdm(range(epochs)):
            loss_train, epoch_train = train_step(
                model, train_data, input_dim,
                sequence_length, loss, optimizer, writer, epoch, device
            )
            loss_val, epoch_val = val_step(
                model, val_data, input_dim,
                sequence_length, loss, writer, epoch, device
            )

            train_values[epoch_train] = loss_train
            val_values[epoch_val] = loss_val

        # save final gradients heatmap
        save_heatmap(model, 'gradients', initialization)

        # save model
        save_model(model, name)

        values[initialization] = {"train": train_values, "val": val_values}

    # plot loss values for each initialization
    plot_loss(values)

    return None


if __name__ == "__main__":
    main()
