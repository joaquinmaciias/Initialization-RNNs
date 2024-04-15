# deep learning libraries
import torch

# other libraries
import math
from typing import Any

import src.initializations as init

class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # TODO

        _, sequence, _ = inputs.size()

        # RNN forward pass
        # Squeeze the hidden state
        hidden_state: torch.Tensor = h0.squeeze(0)

        # List output
        outputs: list[torch.Tensor] = []

        # Create a for loop to iterate over the sequence
        for t in range(sequence):
            # Update the hidden state
            hidden_state = torch.matmul(inputs[:, t, :], weight_ih.t()) +\
                 bias_ih + torch.matmul(hidden_state, weight_hh.t()) + bias_hh

            # Apply ReLU
            hidden_state = torch.relu(hidden_state)

            h = hidden_state.unsqueeze(0)
            output = hidden_state.unsqueeze(1)

            outputs.append(output)

        outputs_tensor: torch.Tensor = torch.cat(outputs, dim=1)

        # Guardamos los elementos para usarlos en el backward
        ctx.save_for_backward(
            inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, outputs_tensor
        )
        return outputs_tensor, h

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # TODO
        # Get the saved data from the forward pass
        inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, outputs = ctx.saved_tensors

        # Initialize gradients
        g_weight_ih = torch.zeros_like(weight_ih)
        g_weight_hh = torch.zeros_like(weight_hh)
        g_bias_ih = torch.zeros_like(bias_ih)
        g_bias_hh = torch.zeros_like(bias_hh)

        _, sequence, _ = inputs.size()

        # g inputs
        g_inputs = []

        g_h = grad_hn.squeeze(0)

        # Backward loops
        for i in range(sequence-1, -1, -1):

            # backwards relu
            g_h += grad_output[:, i, :]
            g_h *= (outputs[:, i, :] > 0).float()

            # Weights grad
            g_weight_ih += torch.matmul(g_h.t(), inputs[:, i, :])
            if i == 0:
                h_ant = h0.squeeze(0)
            else:
                h_ant = outputs[:, i-1]

            g_weight_hh += torch.mm(g_h.t(), h_ant)

            # Inuts grad
            g_input = torch.matmul(g_h, weight_ih)
            g_input = g_input.unsqueeze(1)
            g_inputs.append(g_input)

            # H grad
            g_h = torch.matmul(g_h, weight_hh)

            # Bias grad
            g_bias_ih += g_h.sum(dim=0)
            g_bias_hh += g_h.sum(dim=0)

        # Turn list around
        g_inputs.reverse()

        # Concatenate outputs along the sequence dimension
        g_inputs_tensor: torch.Tensor = torch.cat(g_inputs, dim=1)

        # Add dimension
        g_h0 = g_h.unsqueeze(0)

        return g_inputs_tensor, g_h0, g_weight_ih, g_weight_hh, g_bias_ih, g_bias_hh


class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int, initialization: str):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
       
        # init parameters corectly
        self.reset_parameters(initialization)

        self.fn = RNNFunction.apply
        self.initialization = initialization
        

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self, initialization) -> None:
        """
        This method initializes the parameters in the correct way.
        """
        for weight in self.parameters():
            shape = weight.size()
            if len(shape) > 1: 
                for weight in self.parameters():
                    if initialization == "zeros":
                        init.zeros_initialization(shape)
                    elif initialization == "identity":
                        init.identity_initialization(shape)
                    elif initialization == "identity_001":
                        init.identity_001_initialization(shape)
                    elif initialization == "constant":
                        init.constant_initialization(shape)
                    elif initialization == "random_normal":
                        init.random_normal_initialization(shape)
                    elif initialization == "random_uniform":
                        init.random_uniform_initialization(shape)
                    elif initialization == "truncated_normal":
                        init.truncated_normal_initialization(shape)
                    elif initialization == "xavier":
                        init.xavier_initialization(shape)
                    elif initialization == "normalized_xavier":
                        init.normalized_xavier_initialization(shape)
                    elif initialization == "kaiming":
                        init.kaiming_initialization(shape)
                    elif initialization == "orthogonal":
                        init.orthogonal_initialization(shape)
                    else:
                        raise ValueError("Invalid initialization method.")

        return None


class MyModel(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, initialization: str) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_dim: hidden size of the RNN layers
        """

        # TODO
        super().__init__()

        self.rnn = RNN(24, hidden_dim, initialization)
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
