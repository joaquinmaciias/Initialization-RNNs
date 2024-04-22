
# VISUALIZATION'S CREATOR

# Importing necessary libraries
import seaborn as sns
import os
from utils import read_data
import re
import matplotlib.pyplot as plt
import pandas as pd

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


def histogram(names, training, validation):
    """
    This function creates a histogram plot using the training and validation accuracies.

    Args:
    names: list, list of model names
    training: list, list of training accuracies
    validation: list, list of validation accuracies
    """

    # Create a DataFrame with the data
    data = pd.DataFrame({
        "Model": names,
        "Training Accuracy": training,
        "Validation Accuracy": validation
    })

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="Model", y="Training Accuracy", color="blue", label="Training Accuracy")
    sns.barplot(data=data, x="Model", y="Validation Accuracy", color="red", label="Validation Accuracy")
    plt.xlabel("Model")
    # Add degree rotation to the x-axis labels
    plt.xticks(rotation=22)
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    # Save the plot

    # Create the plots folder
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig("plots/accuracy_validation_histogram.png")



def extract_init_name(file_name):
    # Use regular expression to extract initialization name
    match = re.search(r'model2_init_([a-zA-Z]+(?:_[a-zA-Z]+)?)_', file_name)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def create_visualizations(data_dir):
    # Initialize lists to store data from all initializations
    all_train_loss = []
    all_train_accuracy = []
    all_val_loss = []
    all_val_accuracy = []
    legend_labels = []  # Store legend labels

    # Get list of file names
    try:
        file_names = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{data_dir}' not found.")
        return
    
    print(file_names)

    for file_name in file_names:
        # Read the data
        file_path = os.path.join(data_dir, file_name)
        data = pd.read_csv(file_path)

        # Extract data
        all_train_loss.append(data['train_loss'])
        all_train_accuracy.append(data['train_accuracy'])
        all_val_loss.append(data['val_loss'])
        all_val_accuracy.append(data['val_accuracy'])
        
        # Extract initialization name
        init_name = extract_init_name(file_name)
        legend_labels.append(init_name)  # Add initialization name to legend labels

    # Combine data from all initializations
    combined_train_loss = pd.concat(all_train_loss, axis=1)
    combined_train_accuracy = pd.concat(all_train_accuracy, axis=1)
    combined_val_loss = pd.concat(all_val_loss, axis=1)
    combined_val_accuracy = pd.concat(all_val_accuracy, axis=1)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            if i == 0 and j == 0:
                data = combined_train_loss
                title = 'Train Loss'
                ylabel = 'Loss'
            elif i == 0 and j == 1:
                data = combined_train_accuracy
                title = 'Train Accuracy'
                ylabel = 'Accuracy'
            elif i == 1 and j == 0:
                data = combined_val_loss
                title = 'Validation Loss'
                ylabel = 'Loss'
            else:
                data = combined_val_accuracy
                title = 'Validation Accuracy'
                ylabel = 'Accuracy'
                
            for k in range(data.shape[1]):
                ax.plot(data.iloc[:, k], label=legend_labels[k])  # Use initialization names as legend labels

            ax.set_title(title)
            ax.set_xlabel('Epochs', fontsize='small')
            ax.set_ylabel(ylabel, fontsize='small')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set x-axis to display only integers
            ax.legend(fontsize='small')  # Set legend font size to 'small'

    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Increase separation between plots
    plt.tight_layout()
    plt.savefig('plots/total_visualizations2.png') 
    plt.show()



# Example of use
if __name__ == '__main__':

    # Call the function with the directory path
    create_visualizations('runs/models_data_model2/')   

    data_path = "runs/models_data_model2/"

    files = os.listdir(data_path)

    model_names = []
    training_accuracy = []
    validation_accuracy = []

    for file in files:
        name_orig = file.split("_")

        # We only keep from index 2 until lr not included
        
        name = ""

        for i in range(2, len(name_orig)):
            if name_orig[i] == "lr":
                break
            name += name_orig[i] + "_"

        data = read_data(data_path + file)

        model_names.append(name[:-1])
        training_accuracy.append(data[0]["accuracy"][-1])
        validation_accuracy.append(data[1]["accuracy"][-1])

    histogram(model_names, training_accuracy, validation_accuracy)




