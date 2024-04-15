
# Deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# Other libraries
from typing import Final

# Own modules
try:
    from src.data import load_data
    from src.utils import set_seed, load_model
    from src.train_functions import t_step
except:
    from data import load_data
    from utils import load_model
    from train_functions import t_step
    from main_tmp import SEED

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name) -> None:
    """
    This function is the main program.
    """
    # Load test data
    test_loader: DataLoader
    _, _, test_loader, mean, std = load_data(DATA_PATH, num_workers=4, drop_last=True)

    # Load model
    model = load_model(name).to(device)

    # Set model to evaluation mode
    model.eval()

    # Evaluate model with tstep and get mean MAE score
    mae_score = t_step(model, test_loader, mean, std, device)

    return mae_score


# if __name__ == "__main__":

#     mae = main("best_model")
#     print(f"MAE Score: {mae}")