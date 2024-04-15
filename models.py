# deep learning libraries
import torch

# other libraries
import math
from typing import Any

import src.initializations as init

class MyModel(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, initialization: str) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_dim: hidden size of the RNN layers
        """

        # TODO
        super().__init__()

        self.rnn = torch.nn.RNN(24, hidden_dim, 1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 24)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """
        batch_size = inputs.size(0)
        h0 = torch.zeros(1, batch_size, self.rnn.hidden_size)
        
        # Pass the input through the RNN
        rnn_output, _ = self.rnn(inputs,h0)

        # Apply dropout
        rnn_output = self.dropout(rnn_output)

        # Pass the RNN output through the fully connected layer
        output = self.fc(rnn_output[:, -1, :])

        return output
