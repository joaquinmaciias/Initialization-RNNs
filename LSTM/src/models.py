import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self,pretrained_model,input_size, hidden_size, output_size, num_layers=1):
        super(MyModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Obtain embedding layer from pretrained model
        self.embedding = pretrained_model.embedding
        self.embedding_dim = pretrained_model.embedding_dim
        self.context_size = pretrained_model.context_size
        
        # LSTM layer
        # PENDIENTE: Preguntar sobre input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Classifier layer
        self.fc = nn.Linear(hidden_size*self.context_size, output_size)
        
    def forward(self, x, h0 = None, c0 = None):

        

        x = self.embedding(x)

        if h0 is None and c0 is None:
            outputs, hn = self.lstm(x)
        else:
            outputs, hn = self.lstm(x, (h0, c0))

        outputs = outputs.reshape(-1, self.hidden_size*self.context_size)

        out = self.fc(outputs)

        # We view to get a 2D vector of shape (batch_size, output_size)
        return out