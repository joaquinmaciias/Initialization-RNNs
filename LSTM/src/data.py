# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# other libraries
import os

import numpy as np
import music21
import muspy
import re

class MusicDataset(Dataset):

    def __init__(self, inputs, targets):
        self.data = [(inputs[i], targets[i]) for i in range(len(inputs))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        context, target = self.data[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return context_tensor, target_tensor.squeeze()


def load_data(data_path: str, 
              context_size: int = 6,
              batch_size: int = 64,
              shuffle: bool = True,
              drop_last: bool = True,
              num_workers: int = 0, 
              train_pct: float = 0.8) -> DataLoader:
    
    '''
    This method returns Dataloader of the chosen dataset.

    '''

    # We check if the Model_Data folder exists. If not, we create it.

    if not os.path.exists(data_path + "Model_Data/"):
        os.makedirs(data_path + "Model_Data/")
        print("Downloading data...")
        # We download the data
        train, test = download_data(data_path, train_pct)

    else:
        # We retrieve the data from the folder
        train_path = data_path + "Model_Data/train/"

        print("Loading data...")

        train = []

        for file in os.listdir(train_path):
                
            if file.endswith(".txt"):

                notes = []

                with open(train_path + file, "r") as f:

                    for line in f:

                        line = int(line.strip())

                        notes.append(line)

                train.append(notes)
        
        test_path = data_path + "Model_Data/test/"

        test = []

        for file in os.listdir(test_path):
                
            if file.endswith(".txt"):

                notes = []

                with open(test_path + file, "r") as f:

                    for line in f:

                        notes.append(int(line.strip()))

                test.append(notes)

    # Once we have the data, we need to process it

    # Data are already a number from 0 to 387

    # Start token: 388
    # End token: 389
    # Pad token: 390

    # We will now create the sequences

    print(f'Data loaded successfully with {len(train)} training samples and {len(test)} testing samples.')

    sequences_tr, targets_tr = process_data(train, context_size, start_token=388, end_token=389, pad_token=390)
    sequences_ts, targets_ts = process_data(test, context_size, start_token=388, end_token=389, pad_token=390)

    print(f'Data processed successfully with {len(sequences_tr)} training sequences and {len(sequences_ts)} testing sequences.')

    # We create the datasets

    train_dataset = MusicDataset(sequences_tr, targets_tr)
    test_dataset = MusicDataset(sequences_ts, targets_ts)

    # We create the dataloaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, test_loader


def download_data(path: str = "data/", train_pct: float = 0.8) -> list:

    # For every file in the folder, we will extract the notes and save them in a .txt file

    # We check how many files ended in .midi we have

    files = []

    for file in os.listdir(path + "MIDI_Files/"):

        if file.endswith(".midi"):

            music = muspy.read_midi(path + "MIDI_Files/" + file)

            events = muspy.to_event_representation(music, encode_velocity=True)

            files.append(events)

    # We divide the data into training and testing randomly

    # We use random_split

    train, test = random_split(files, [int(len(files) * train_pct), len(files) - int(len(files) * train_pct)])

    train_clean = []
    test_clean = []
    # We save the data in the Model_Data folder

    if not os.path.exists(path + "Model_Data/train/"):
        os.makedirs(path + "Model_Data/train/")

        for i in range(len(train)):

            file = []

            with open(path + "Model_Data/train/" + str(i) + ".txt", "w") as f:

                for note in train[i]:

                    if len(note) > 1:

                        print("ERROR")

                    file.append(note[0])

                    f.write(str(note[0]) + "\n")
            
            train_clean.append(file)

    if not os.path.exists(path + "Model_Data/test/"):
        os.makedirs(path + "Model_Data/test/")

        for i in range(len(test)):

            file = []

            with open(path + "Model_Data/test/" + str(i) + ".txt", "w") as f:

                for note in test[i]:

                    if len(note) > 1:
                            
                        print("ERROR")
                    
                    file.append(note[0])
                    
                    f.write(str(note[0]) + "\n")

            test_clean.append(file)
    
    return train_clean, test_clean


def process_data(data: list, context_size: int, start_token: int, end_token: int, pad_token: int) -> list:
    
    # TODO

    # We will create sequences and targets (next note) for the data

    sequences = []
    targets = []

    for song in data:

        song = [start_token] + song + [end_token]
            
        for i in range(len(song) - context_size):

            sequences.append(song[i:i + context_size])
            targets.append(song[i + context_size])

    # We will pad the sequences

    for i in range(len(sequences)):
            
        sequences[i] = sequences[i] + [pad_token] * (context_size - len(sequences[i]))

    return sequences, targets

   



    pass


    
if __name__ == "__main__":

    out = load_data("data/")

    print(out)

    pass


