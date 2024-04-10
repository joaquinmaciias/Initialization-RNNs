# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
import pandas as pd
import music21


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """

    # TODO
    for param in model.parameters():
        param.data = param.data.double()
    
    return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def midi_to_notes(midi: str) -> pd.DataFrame:
    """
    This function converts a midi file to a list of notes.

    Args:
        midi: path to the midi file.

    Returns:
        list of notes.
    """

    # TODO

    # Load midi file

    midi_data = music21.converter.parse(midi)

    # Get the instrument parts

    instrument = music21.instrument.partitionByInstrument(midi_data)

    # Get the notes

    noteFilter = music21.stream.filters.ClassFilter('Note')

    notes_to_parse = instrument.parts[0].recurse()

    # Create a list of notes

    notes = []

    for element in notes_to_parse:

        word = ""
            
        if isinstance(element, music21.note.Note):

            '''
            Word format:
            - pitch
            - octave
            - duration
            '''

            word = "{}_{}_{}".format(element.pitch.name, element.pitch.octave, element.duration.type)
            notes.append(word)

        elif isinstance(element, music21.chord.Chord):
                
                # Chords will be saved by their name and duration

                word = "{}_{}".format(element.fullName, element.duration.type)

                print(word)
                notes.append(word)
            

    return notes
        

def predict_sequence(model: torch.nn.Module,
                         note_to_idx: dict,
                         idx_to_note: dict,
                         sequence_size: int) -> list:
        """
        This function predicts a musical sequence.

        Args:
            model: pytorch model.
            note_to_idx: dictionary mapping notes to indices.
            idx_to_note: dictionary mapping indices to notes.
            sequence_size: size of the sequence to predict.

        Returns:
            list of predicted notes.
        """

        # Create a list to store the predicted notes

        predicted_notes = []

        # Create a random sequence to start the prediction

        sequence = torch.randint(0, len(note_to_idx), (1, sequence_size)).long()

        # Iterate over the sequence size

        for i in range(sequence_size):

            # Predict the next note

            output, _ = model(sequence)

            # Get the last output

            last_output = output[:, -1, :]

            # Get the index of the maximum value

            max_index = torch.argmax(last_output, dim=1)

            # Append the index to the sequence

            sequence = torch.cat((sequence, max_index.view(1, 1)), dim=1)

            # Append the note to the list

            predicted_notes.append(idx_to_note[max_index.item()])

        return predicted_notes