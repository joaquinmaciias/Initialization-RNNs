import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random

try:
    from src.data import load_data
    from src.utils import save_model, train_model, save_model
except:
    from data import load_data
    from utils import save_model, train_model, save_model
    from train_functions import train_step, val_step
    import parameters as p
    # import main_tmp as main




class PretrainedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(PretrainedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.embedding_dim = embedding_dim
        self.context_size = context_size

    def forward(self, context_idxs):
        embedded = torch.sum(self.embedding(context_idxs), dim=1)
        out = self.linear(embedded)

        return out
    
def train_embedding(train, val, test, context_size, embedding_dim, epochs, learning_rate, patience, vocab_size, device):

    model = PretrainedModel(vocab_size, embedding_dim, context_size)

    model.to(device)

    print(f"Training model in {device}...")

    optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables for Early Stopping
    best_loss = float('inf')
    epochs_no_improve = 0

    model = model.to(device)
    criterion = criterion.to(device)

    writer = None

    for epoch in range(epochs):
        
        loss_train, accuracy_train = train_step(model, train, criterion, optimzer, writer, epoch, device)
        
        loss_val, accuracy_val = val_step(model, val, criterion,writer, epoch, device)

        if loss_val < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch} with best validation loss of {best_loss}")
            break

    
    # TODO Check test performance

    # save model
    runs_folder = "runs/embeddings"  # Folder to save models
    model_filename = "music" + f"_embedding_dim_{embedding_dim}_context_size_{context_size}.pth"
    model_path = os.path.join(runs_folder, model_filename)  # Full path to the model
    # Save the entire model

    save_model(model.to(torch.device("cpu")), model_path)
    
    return model


if __name__ == "__main__":
    # Setting random seeds for reproducibility

    path = "data/"

    context_size = p.context_size
    embedding_dim = p.embedding_dim
    epochs = p.epochs_emb
    learning_rate = p.learning_rate_emb
    patience = p.patience
    vocab_size = p.vocab_size
    batch_size = p.batch_size

    train, val, test = load_data(path, context_size, batch_size)
   
    print("Data loaded and tokenized.")

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Starting training...")
    pt_model = train_embedding(train, val, test, context_size, embedding_dim, epochs, learning_rate, patience, vocab_size, device)
    print("Training finished.")
    print("Model saved.")

    