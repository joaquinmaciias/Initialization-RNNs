import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def extract_init_name(file_name):
    # Use regular expression to extract initialization name
    match = re.search(r'model_init_([a-zA-Z]+(?:_[a-zA-Z]+)?)_', file_name)
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
    plt.savefig('runs/images/total_visualizations2.png') 
    plt.show()

# Call the function with the directory path
create_visualizations('runs')
