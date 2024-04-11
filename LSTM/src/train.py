# Deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Final

# Own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model

# Static variables
DATA_PATH: Final[str] = "data"

# Set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(
    epochs: int = 50,
    lr: float = 8e-4,
    batch_size: int = 64,
    hidden_size: int = 256,
    
) -> None:
    """
    This function is the main program for training.
    """
    # Empty nohup file
    open("nohup.out", "w").close()

    # Load training and validation data
    train_loader: DataLoader
    val_loader: DataLoader
    train_loader, val_loader, _, mean, std = load_data(
        DATA_PATH, num_workers=4, batch_size=batch_size, drop_last=True
    )

    # define name and writer
    name: str = (
        f"model_lr_{lr}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # Define model
    model = MyModel(hidden_size=hidden_size).to(device)

    # Define loss function and optimizer
    loss: torch.nn.Module = torch.nn.L1Loss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(epochs)):

        # Implement training loop
        train_step(model=model,train_data=train_loader,mean=mean,std=std,
            loss=loss, optimizer=optimizer,writer=writer,epoch=epoch,device=device,
        )

        # Implement validation loop
        val_step(model=model, val_data=val_loader, mean=mean,std=std,
                 loss=loss,writer=writer,epoch=epoch, device=device,scheduler=None,
        )


    # Save model
    save_model(model, "best_model")



if __name__ == "__main__":
    main()
