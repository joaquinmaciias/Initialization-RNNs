# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# other libraries
import os
import re
import numpy as np

class NewsDataset():

    def __init__(self, inputs: str, targets: int, vocab_to_int: dict, int_to_vocab: dict, vocab_size: int, start, end, padding):
        
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.vocab_size = vocab_size

        self.data = []

        for i in range(len(inputs)):
            text = inputs[i]
            tokens_int = [self.vocab_to_int[word] for word in text]
            self.data.append((tokens_int, targets[i]))
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            
        context, target = self.data[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return context_tensor, target_tensor.squeeze()


def load_data(data_path: str, 
              batch_size: int = 64,
              shuffle: bool = True,
              drop_last: bool = True,
              num_workers: int = 0, 
              train_pct: float = 0.8) -> DataLoader:
    
    '''
    This method returns Dataloader of the chosen dataset.

    '''

    start = "<\S>"
    end = "<\D>"
    padding = "<\P>"

    # We rettrieve the data from the folder

    data = pd.read_csv(data_path, encoding='latin-1', header=None)

    # We add a column with 0, 1, 2 for the different classes

    sentiment = {'negative': 0, 'neutral': 1, 'positive': 2}

    data[0] = [sentiment[i] for i in data[0]]

    # We tokenize the data
    inputs = []
    labels = list(data[0])
    longest = 0
    shortest = 1000

    for i in range(len(data)):
        text = data[1][i]
        tokens = tokenize_data(text, start, end)
        inputs.append(tokens)
        if len(tokens) > longest:
            longest = len(tokens)
        
        if len(tokens) < shortest:
            shortest = len(tokens)


    print(f"Shortest text: {shortest}")

    print(f"Longest text: {longest}")

    # padding the data

    for i in range(len(inputs)):
        while len(inputs[i]) < longest:
            inputs[i].append(padding)

    vocab_to_int, int_to_vocab, vocab_size = create_lookup_tables(inputs)

    # Generate Dataset randomly dividing the data into training and testing

    # We shuffle the data
    indices = list(range(len(inputs)))
    np.random.shuffle(indices)
    inputs = [inputs[i] for i in indices]

    train_size = int(len(inputs) * train_pct)
    train_inputs = inputs[:train_size]
    test_inputs = inputs[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    # We now divide the validation from the training

    val_size = int(len(train_inputs) * (1-train_pct))
    val_inputs = train_inputs[:val_size]
    train_inputs = train_inputs[val_size:]
    val_labels = train_labels[:val_size]
    train_labels = train_labels[val_size:]

    train_dataset = NewsDataset(train_inputs, train_labels, vocab_to_int, int_to_vocab, vocab_size, start, end, padding)
    val_dataset = NewsDataset(val_inputs, val_labels, vocab_to_int, int_to_vocab, vocab_size, start, end, padding)
    test_dataset = NewsDataset(test_inputs, test_labels, vocab_to_int, int_to_vocab, vocab_size, start, end, padding)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, val_dataloader, test_loader


def create_lookup_tables(tokens):
    """
    Create lookup tables for vocabulary
    """
    vocab = set([word for text in tokens for word in text])
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for i, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab, len(vocab)



def tokenize_data(text, start, end):

    # We lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # split the text
    text = text.split()

    return [start] + text + [end]


if __name__ == "__main__":
    data_path = "data/news.csv"
    yes = load_data(data_path)
    print(yes)
    

    








    

    

    