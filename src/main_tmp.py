'''
Archivo provisional de modelo LSTM para análisis de música y reproducción de melodías
'''

# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import model
from models import LSTM
# O hacerlo con el modelo de torch.nn

# Import train, validation and test function
from src.train_functions import train_step , val_step, t_step
# from test import test_model

# Import loss functions from which to choose
from torch.nn import CrossEntropyLoss, MSELoss

# Import predict function
from utils import predict_sequence, save_model

# Import evaluate function
from evaluate import main as main_eval

# Import load_data function
from data import load_data


# Paths
path = '/data/'

# Load data parameters
save_path = ''
sequence_size = 6
batch_size = 64
shuffle = True
drop_last = True
num_workers = 0

# Load data
print("Loading data...")

# PENDIENTE ENRIQUE : Sacar val_dataloader también
tr_dataloader, ts_dataloader= load_data(save_path,
                                        sequence_size,
                                        batch_size,
                                        shuffle,
                                        drop_last,
                                        num_workers)

# Check the length of the dictionary

## PENDIENTE ENRIQUE: Sacar el diccionario de la función load_data

note_to_idx = load_data.note_to_idx
idx_to_note = load_data.idx_to_note
print("Length of note_to_index:", len(note_to_idx))
print("Length of index_to_note:", len(idx_to_note))

# Define the music to vector model
# m2v_model = ''

# Define the LSTM model parameters
input_size = 3
hidden_size = 128
learning_rate = 0.001
epochs = 10
num_layers = 2

## PENDIENTE ENRIQUE: Sacar el número de notas del diccionario
num_notes = len(note_to_idx)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define name and writer

name: str = (
    f"model_lr_{learning_rate}_hs_{hidden_size}_bs_{batch_size}_e_{epochs}"
)
writer: SummaryWriter = SummaryWriter(f"runs/{name}")


# Create the model and migrate to device

# Sol 1 - Define the model with nn
# model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)

# Sol 2 - Define the model with the class
model = LSTM(input_size, hidden_size, num_layers, num_notes, device).to(device)


# Define optimizer and loss function using MAE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = MSELoss()

## PENDIENTE : Posibilidad de implementar un scheduler
## PENDIENTE : Implementar un early stopping

## PENDIENTE ENRIQUE: Sacar mean y std en el load data para los loops de train y val

# Training loop
# for epoch in tqdm(range(epochs)):

    # Implement training loop
    
    # Set the model in training mode
    # model.train()
    # train_step(model=model,train_data=tr_dataloader,mean=mean,std=std,
    #     loss=loss_function, optimizer=optimizer,writer=writer,epoch=epoch,device=device,
    # )

    # Implement validation loop
    # Set the model in evaluation mode
    # model.eval()
    # val_step(model=model, val_data=val_dataloader, mean=mean,std=std,
    #             loss=loss,writer=writer,epoch=epoch, device=device,scheduler=None,
    # )

# Save the model with the save_model function
save_model(model, name)

# Execute the evaluate.py
mae = main_eval(name)

# Predict a musical sequence
pred_melody = predict_sequence(model, note_to_idx, idx_to_note, sequence_size)

