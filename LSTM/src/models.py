import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,pretrained_model,input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Obtain embedding layer from pretrained model
        self.embedding = pretrained_model.embedding
        self.embedding_dim = pretrained_model.embedding_dim
        self.context_size = pretrained_model.context_size
        
        # LSTM layer
        # PENDIENTE: Preguntar sobre input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output linear layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        x = self.emmbedding(x)
        ## PENDIENTE : Comprobar si es necesario hacer el reshape
        x = x.view(x.size(0), self.embedding_dim*self.context_size)

        ####  INICIALIZACIÓN DE H0 Y C0 EN LSTM  ####
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        outputs, _ = self.lstm(x)
        
        return outputs
