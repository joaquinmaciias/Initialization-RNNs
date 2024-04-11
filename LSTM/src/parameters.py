
# Embeddings model parameters

context_size = 6
embedding_dim = 10
epochs_emb = 10
learning_rate_emb = 0.0001
batch_size = 32
patience = 10
vocab_size = 391


# LSTM model parameters
input_size = vocab_size
output_size = 1 # Ya que necesitamos saber cual es la prócima palabra
hidden_size = 128
learning_rate = 0.001
epochs = 10
num_layers = 2