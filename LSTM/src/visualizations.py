
# VISUALIZATION'S CREATOR

# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Function to create a double line plot using epochs and loss values
def line_plot(x_values, y1_values, y2_values, x_label, y_label, title, y1_label, y2_label, vis_name):
    """
    This function creates a double line plot using epochs and loss values.

    Args:
    x_values: list, list of x values
    y1_values: list, list of y1 values
    y2_values: list, list of y2 values
    x_label: str, x axis label
    y_label: str, y axis label
    title: str, plot title
    y1_label: str, y1 axis label
    y2_label: str, y2 axis label

    Returns:
    None
    """
        
    # Create a DataFrame with the data
    data = pd.DataFrame({
        "Epochs": x_values,
        y1_label: y1_values,
        y2_label: y2_values
    })

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epochs", y=y1_label, label=y1_label)
    sns.lineplot(data=data, x="Epochs", y=y2_label, label=y2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Save the plot

    # Create the plots folder
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{vis_name}.png")



# Example of use
if __name__ == '__main__':

    # Loss and MAE lists for the visualization
    train_loss_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    val_loss_list = [0.2, 0.3, 0.4, 0.5, 0.6]

    # Create the visualization - Loss
    line_plot(
        x_values=range(5),
        y1_values=train_loss_list,
        y2_values=val_loss_list,
        x_label="Epochs",
        y_label="Loss",
        title="Loss Plot",
        y1_label="Train Loss",
        y2_label="Validation Loss",
        vis_name="example_loss"
    )
