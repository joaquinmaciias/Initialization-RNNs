import os
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

def save_model(model, model_path: str):
    """Save the trained SkipGram model to a file, creating the directory if it does not exist.

    Args:
        model: The trained SkipGram model.
        model_path: The path to save the model file, including directory and filename.

    Returns:
        The path where the model was saved.
    """
    # Extract the directory path from the model_path
    directory = os.path.dirname(model_path)
    
    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    return model_path


def train_model(model, train_loader, test_loader, epochs, learning_rate, device, print_every=1000, patience=10):
    """
    Train a Pytorch model.

    Args:
        model (torch.nn.Module): Pytorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        epochs (int): The number of epochs to train the model.
        device (str): device where to train the model.
        learning_rate (float): The learning rate for the optimizer.
        print_every (int): Frequency of epochs to print training and test loss.
        patience (int): The number of epochs to wait for improvement on the test loss before stopping training early.
    """

    # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
    optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables for Early Stopping
    best_loss = float('inf')
    epochs_no_improve = 0

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        # Train model
        total_train_loss = 0

        model.train()

        for inputs, targets in train_loader:
            # Move input and target tensors to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimzer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss

            loss = criterion(outputs, targets)

            # Backward pass

            loss.backward()

            # Update the weights

            optimzer.step()

            # Update the total loss

            total_train_loss += loss.item()



        # Evaluate on test set
        total_test_loss = 0

        with torch.no_grad():
            model.eval()
            for inputs, targets in test_loader:
                # Move input and target tensors to the device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute the loss

                loss = criterion(outputs, targets)

                # Update the total loss

                total_test_loss += loss.item()

        # Print losses
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Training Loss: {total_train_loss / len(train_loader)}, Test Loss: {total_test_loss / len(test_loader)}")

        # Check for Early Stopping
        if total_test_loss < best_loss:
            best_loss = total_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. No improvement in test loss for {patience} consecutive epochs.")
                break  # Stop training early