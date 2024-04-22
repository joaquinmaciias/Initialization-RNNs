
# Embeddings model parameters

context_size = 32
embedding_dim = 32
epochs_emb = 15
learning_rate_emb = 0.0001
batch_size = 64
patience = 5
vocab_size = 359


# LSTM model parameters
input_size = embedding_dim
output_size = vocab_size
hidden_size = 360
learning_rate = 0.0001
epochs = 30
num_layers = 1