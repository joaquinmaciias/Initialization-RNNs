import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def save_heatmap(model: torch.nn.Module, folder: str, initialization: str):
    """
    This function creates a heat map of the weights or gradients of
    the model and saves it in a specific initialization folder.

    Args:
        model (torch.nn.Module): Model to analyze.
        folder (str): Folder to save the heat map.
        initialization (str): Initialization method of the model.
    """
    # Define the directory for the specific initialization
    directory = os.path.join(os.getcwd(), folder, initialization)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for name, param in model.named_parameters():

        if name in ['rnn.weight_hh_l0', 'rnn.weight_hh_l1']:

            data: Optional[torch.Tensor] = None
            if folder == 'weights':
                data = param
            elif folder == 'gradients' and param.grad is not None:
                data = param.grad

            if data is not None:

                # Create a heatmap with a fixed value range
                plt.figure(figsize=(10, 8))
                heatmap_data = data.cpu().detach().numpy()
                sns.heatmap(
                    heatmap_data, annot=False, fmt='f',
                    cmap='coolwarm', center=0, cbar=False
                    )
                plt.title(f'Heatmap of {folder} {name} of {initialization}')

                # Save the heatmap as a PNG in the specific directory
                output_path = os.path.join(
                    directory, f'{name}_{folder}_{initialization}.png'
                    )
                plt.savefig(output_path)
                plt.close()


def plot_loss(values: dict):
    """
    This function plots two different png files: one with the loss
    values of the training across different initializations,
    and another with the loss values of the validation across different initializations.

    Args:
        values (dict): Dictionary with the loss values of the training and validation
        across different initializations.
    """

    # Verfies if the 'plots' folder exists, if not, creates it
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Initialize the figure for the training losses
    plt.figure(figsize=(10, 8))
    # Plots the training losses of all the initializations
    for key in values:
        train_values = values[key]["train"]
        plt.plot(list(train_values.keys()), list(train_values.values()), label=key)

    plt.xlim(left=0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Evolution Across Initializations")
    plt.legend()
    plt.savefig("plots/train_loss.png")
    plt.close()

    # Initialize the figure for the validation losses
    plt.figure(figsize=(10, 8))

    # Plots the validation losses of all the initializations
    for key in values:
        val_values = values[key]["val"]
        plt.plot(list(val_values.keys()), list(val_values.values()), label=key)

    plt.xlim(left=0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Evolution Across Initializations")
    plt.legend()
    plt.savefig("plots/val_loss.png")
    plt.close()


def accuracy_histogram(accuracies: dict):
    """
    This function plots a histogram of the accuracies across different initializations.
    And saves it in a PNG file and plots folder.

    Args:
        accuracies (dict): Dictionary with the accuracies across different initialization
    """

    # Save in the 'plots' folder
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Initialize figure
    plt.figure(figsize=(10, 8))

    # Generate a histogram of the accuracies across different initializations
    x: list = list(accuracies.keys())
    y: list = list(accuracies.values())
    plt.bar(x, y)
    plt.xticks(rotation=30)
    plt.xlabel("Initializations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Initializations")
    plt.savefig("plots/accuracy_histogram.png")
    plt.close()
