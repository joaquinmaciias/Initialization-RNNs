# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# other libraries
import os

import numpy as np
import music21
import muspy


def load_data(data_path: str, 
              sequence_size: int = 6,
              batch_size: int = 64,
              shuffle: bool = True,
              drop_last: bool = True,
              num_workers: int = 0) -> DataLoader:
    
    '''
    This method returns Dataloader of the chosen dataset.

    '''

    # We check if the Model_Data folder exists. If not, we create it.

    if not os.path.exists(data_path + "Model_Data/"):

        # PENDIENTE ENRIQUE: Este es un cambio que he hecho yo para que no de error

        # Previo:
        # os.makedirs(data_path + "Model_Data/")
        
        os.makedirs(data_path + "Model_Data/train/")
        os.makedirs(data_path + "Model_Data/test/")

        # We download the data
        train, test = download_data(data_path)

    else:
        # We retrieve the data from the folder
        train_path = data_path + "Model_Data/train/"

        train = []

        for file in os.listdir(train_path):
                
            if file.endswith(".txt"):

                notes = []

                with open(train_path + file, "r") as f:

                    for line in f:

                        notes.append(line.strip())

                train.append(notes)
        
        test_path = data_path + "Model_Data/test/"

        test = []

        for file in os.listdir(test_path):
                
            if file.endswith(".txt"):

                notes = []

                with open(test_path + file, "r") as f:

                    for line in f:

                        notes.append(line.strip())

                test.append(notes)

    # Once we have the data, we need to process it

    # Data are already a number from 0 to 387

    # Start token: 388
    # End token: 389
    # Pad token: 390

    # We will now create the sequences

    sequences_tr = process_data(train, sequence_size, start_token=388, end_token=389, pad_token=390)
    sequences_ts = process_data(test, sequence_size, start_token=388, end_token=389, pad_token=390)

    # We create the dataloader

    dataset_tr = torch.tensor(sequences_tr, dtype=torch.long)
    dataset_ts = torch.tensor(sequences_ts, dtype=torch.long)

    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    dataloader_ts = DataLoader(dataset_ts, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader_tr, dataloader_ts


def download_data(path: str = "data/") -> list:

    # For every file in the folder, we will extract the notes and save them in a .txt file

    # We check how many files ended in .midi we have

    files = []

    for file in os.listdir(path + "MIDI_Files/"):

        if file.endswith(".midi"):

            music = muspy.read_midi(path + "MIDI_Files/" + file)

            events = muspy.to_event_representation(music, encode_velocity=True)

            files.append(events)

    # We divide the data into training and testing randomly

    np.random.shuffle(files)

    train = files[:int(0.8 * len(files))]
    test = files[int(0.8 * len(files)):]

    # We save the data in the Model_Data folder

    if not os.path.exists(path + "Model_Data/train/"):
        os.makedirs(path + "Model_Data/train/")

        for i in range(len(train)):

            with open(path + "Model_Data/train/" + str(i) + ".txt", "w") as f:

                for note in train[i]:

                    f.write(str(note) + "\n")

    if not os.path.exists(path + "Model_Data/test/"):
        os.makedirs(path + "Model_Data/test/")

        for i in range(len(test)):

            with open(path + "Model_Data/test/" + str(i) + ".txt", "w") as f:

                for note in test[i]:

                    f.write(str(note) + "\n")
    
    return train, test


def process_data(data: list, sequence_size: int, start_token: int, end_token: int, pad_token: int) -> list:
    
    # TODO
    pass


    
if __name__ == "__main__":

    out = load_data("data/", 6, 64, True, True, 0)
    print(out)
    pass


