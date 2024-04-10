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
    

if __name__ == "__main__":
    # Setting random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Load and preprocess data
    infile = "data/"
    context_size = 6  # Define the size of the context window
    embedding_dim = 10  # Define the size of the embedding vector
    epochs = 10
    learning_rate = 0.0001
    batch_size = 16
    patience = 10
    vocab_size = 391 # Check muspy doc
    print("Loading data...")
    train, test = load_data(infile)
    print("Data loaded and tokenized.")


    # Initialize CBOW Model
    model = PretrainedModel(vocab_size, embedding_dim, context_size)

    # Check if GPU is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training model in {device}...")

    train_model(model, train, test, epochs, learning_rate, device, print_every=10, patience=patience)
    print("Model finished.")

    runs_folder = "runs/embeddings"  # Folder to save models
    model_filename = "music" + f"_embedding_dim_{embedding_dim}_context_size_{context_size}.pth"
    model_path = os.path.join(runs_folder, model_filename)  # Full path to the model
    # Save the entire model

    save_model(model.to(torch.device("cpu")), model_path)