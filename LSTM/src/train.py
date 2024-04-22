# Import libraries
import os
import numpy as np
# import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import model
from models import MyModel

# Import train, validation and test function
from train_functions import train_step , val_step
# from test import test_model

# Import loss functions from which to choose
from torch.nn import CrossEntropyLoss

# Import predict function
from utils import save_model, save_data

# Import load_data function
from data_processing import load_data

# Import embeddings.py
import embeddings as emb

# Import parameters
import parameters as p

# Import initializations
import initializations as init

# Import visualizations
import visualizations as vis


# Paths
path = 'data/'

# Embedding model parameters
batch_size = p.batch_size
context_size = p.context_size
embedding_dim = p.embedding_dim
learning_rate_emb = p.learning_rate_emb
patience = p.patience
vocab_size = p.vocab_size

# Define the LSTM model parameters

# LSTM parameters
input_size = p.input_size
output_size = p.output_size
hidden_size = p.hidden_size
learning_rate = p.learning_rate
epochs = p.epochs
num_layers = p.num_layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":

    
    tr_dataloader, val_dataloader, ts_dataloader = load_data(path, context_size, batch_size)
        

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if "runs/embeddings/" exists

    if not os.path.exists("runs/embeddings/"):

        # Warning message
        print("No se ha encontrado el directorio 'runs/embeddings/'. Se procederá a entrenar el modelo de embeddings.")
        # Execute embeddings.py

        pt_model = emb.train_embedding(tr_dataloader, val_dataloader, ts_dataloader, context_size, embedding_dim, epochs, learning_rate_emb, patience, vocab_size, device)

    else:
        filename = "music" + f"_embedding_dim_{embedding_dim}_context_size_{context_size}.pth"
        print(f"Loading embeddings model from 'runs/embeddings/{filename}'...")
        embeddings_model = torch.load(f"runs/embeddings/{filename}")
        pt_model = emb.PretrainedModel(vocab_size, embedding_dim, context_size)
        model_path = os.path.join("runs/embeddings/", filename)  # Full path to the model
        pt_model.load_state_dict(torch.load(model_path))

    # for param in pt_model.parameters():
    #         param.requires_grad = False


    ### INITIALIZATIONS ###

    # Define initializations
    initializations = [init.constant_initialization, 
                    init.random_normal_initialization, init.random_uniform_initialization,
                    init.zeros_initialization,
                    init.truncated_normal_initialization, init.xavier_initialization, init.normalized_xavier_initialization,
                    init.kaiming_initialization, init.orthogonal_initialization]

    initialization_names = ["constant", "random_normal","random_uniform",
                        "zeros",
                        "truncated_normal", "xavier", "normalized_xavier", "kaiming", "orthogonal"]


    # Define the loop for the initializations

    for initialization in initializations:
        # Print the initialization
        print(f"Training model with : {initialization_names[initializations.index(initialization)]}")

        # Define name and writer
        name: str = (
            f"model_init_{initialization_names[initializations.index(initialization)]}_lr_{learning_rate}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}"
        )
        writer: SummaryWriter = SummaryWriter(f"writer/{name}")

        ## PENDIENTE --> Definir output_size

        # Initialize the weights
        # PENDIENTE -> Comprobar si los shapes son correctos
        # FUENTE: https://discuss.pytorch.org/t/how-to-initialize-weights-bias-of-rnn-lstm-gru/2879/2
        
        weights = {'weight_ih': torch.Tensor(initialization((4*hidden_size,input_size))),
                    'weight_hh': torch.Tensor(initialization((4*hidden_size,hidden_size)))
                    }


        # Create the model and migrate to device
        model = MyModel(pt_model,input_size, hidden_size, output_size, num_layers, weights).to(device)


        # Define optimizer and loss function (MAE will be calculated in the evaluation step)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = CrossEntropyLoss()

        print(f"Training model in {device}...")
        
        # Loss and accuracy lists for the visualization
        metrics_train: dict = {"loss": [], "accuracy": []}
        metrics_val: dict = {"loss": [], "accuracy": []}

        # Training loop
        for epoch in tqdm(range(epochs)):

            # Implement training loop
            train_loss, train_acc = train_step(model=model,train_loader=tr_dataloader,
                loss=loss_function, optimizer=optimizer,writer=writer,epoch=epoch,device=device
            )

            metrics_train["loss"].append(train_loss)
            metrics_train["accuracy"].append(train_acc)

            # Implement validation loop
            val_loss, val_acc  = val_step(model=model, val_loader=val_dataloader, loss=loss_function, writer=writer, epoch=epoch, device=device)

            metrics_val["loss"].append(val_loss)
            metrics_val["accuracy"].append(val_acc)


        path = "runs/models/"
        txt_path = "runs/models_data/"

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        # Save the model
        save_model(model, path +name)

        # Save the data
        save_data(metrics_train, metrics_val, txt_path + name + ".csv")
        

        # Create the visualization - Loss
        vis.line_plot(
            x_values=range(epochs),
            y1_values=metrics_train["loss"],
            y2_values=metrics_val["loss"],
            x_label="Epochs",
            y_label="Loss",
            title=f"Loss Plot - {initialization_names[initializations.index(initialization)]}",
            y1_label="Train Loss",
            y2_label="Validation Loss",
            vis_name=f"{name}_loss"
        )

        # Create the visualization - Accuracy
        vis.line_plot(
            x_values=range(epochs),
            y1_values=metrics_train["accuracy"],
            y2_values=metrics_val["accuracy"],
            x_label="Epochs",
            y_label="Accuracy",
            title=f"Accuracy Plot - {initialization_names[initializations.index(initialization)]}",
            y1_label="Train Accuracy",
            y2_label="Validation Accuracy",
            vis_name=f"{name}_accuracy"
        )


    # Execute the evaluate.py
    # mae = main_eval(name)

    # Predict a musical sequence
    # pred_melody = predict_sequence(model, note_to_idx, idx_to_note, sequence_size)

