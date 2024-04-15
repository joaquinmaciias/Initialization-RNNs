
# Embeddings model parameters

context_size = 16
embedding_dim = 32
epochs_emb = 15
learning_rate_emb = 0.0001
batch_size = 64
patience = 5
vocab_size = 391


# LSTM model parameters
input_size = embedding_dim
output_size = vocab_size
hidden_size = 128
learning_rate = 0.001
epochs = 15
num_layers = 1