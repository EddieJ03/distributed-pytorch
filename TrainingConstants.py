### CONSTANTS ###
seed = 42

lengths = [16,32,48,64,80]

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = max(lengths)  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer , OG: 1e-3
n_embd = 512  # Embedding dimension
n_head = 8 # Number of attention heads
n_layer = 6  # Number of transformer encoder layers
feed_forward = 2048


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 512  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 46  # Output size for the classifier, we have 46 presidents
epochs_CLS = 7 # epochs for classifier training