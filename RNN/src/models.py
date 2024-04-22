# deep learning libraries
import torch

# our libraries
import src.initializations as init


class RNN(torch.nn.Module):
    """
    This class represents a custom RNN model with configurable initialization.
    """

    def __init__(
            self, input_dim: int, hidden_size: int, num_layers: int,
            num_classes: int, initialization: str
            ) -> None:
        """
        Initialize the RNN model.

        Args:
            input_dim (int): Dimension of the input feature.
            hidden_size (int): Number of units in the hidden layer.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of classes in the dataset.
            initialization (str): Method to use for parameter initialization.
        """
        super(RNN, self).__init__()

        # parameters
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initialization = initialization
        self.num_classes = num_classes

        # RNN Layer
        self.rnn = torch.nn.RNN(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
            )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, self.num_classes)

        # Initialize parameters
        self.reset_parameters(initialization)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                        [batch_size, sequence_length, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """

        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)

        # RNN layer
        rnn_output, _ = self.rnn(inputs, h0)

        # Fully connected layer
        output = self.fc(rnn_output[:, -1, :])

        return output

    def reset_parameters(self, initialization: str):
        """
        Initialize the parameters of the RNN and linear layers.
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name or 'fc.weight' in name:
                if initialization == "identity":
                    init.identity_initialization(param)
                elif initialization == "identity_001":
                    init.identity_001_initialization(param)
                elif initialization == "zeros":
                    init.zeros_initialization(param)
                elif initialization == "constant05":
                    init.constant_initialization(param, 0.5)
                elif initialization == "constant_05":
                    init.constant_initialization(param, -0.5)
                elif initialization == "random_normal":
                    init.random_normal_initialization(param)
                elif initialization == "random_uniform":
                    init.random_uniform_initialization(param)
                elif initialization == "truncated_normal":
                    init.truncated_normal_initialization(param)
                elif initialization == "xavier":
                    init.xavier_initialization(param)
                elif initialization == "normalized_xavier":
                    init.normalized_xavier_initialization(param)
                elif initialization == "kaiming":
                    init.kaiming_initialization(param)
                elif initialization == "orthogonal":
                    init.orthogonal_initialization(param)
            elif 'bias' in name or 'fc.bias' in name:  # Bias of RNN and Linear
                torch.nn.init.zeros_(param)
