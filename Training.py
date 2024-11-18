### TRAINING ###
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from TrainingConstants import block_size
from Encoder import Classifier
import os

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    with open(os.path.join(directory, 'train.tsv'), 'r', encoding='utf-16') as file:
        for line in file:
            yield line.strip()


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier: Classifier, data_loader, device):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
            
        accuracy = (100 * total_correct / total_samples)
        
        return accuracy

def train_epoch(data_loader, model: Classifier, optimizer, device):
    # size = len(data_loader.dataset)

    num_batches = len(data_loader)
    model.train()
    train_loss, total_correct, total_samples = 0, 0, 0

    for batch, (X, Y) in enumerate(data_loader):
        X, Y = X.to(device), Y.to(device)


        pred, _ = model(X)

        _, predicted = torch.max(pred.data, 1)
        total_correct += (predicted == Y).sum().item()
        total_samples += Y.size(0)



        loss = F.cross_entropy(input=pred, target=Y)

        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = (100 * total_correct / total_samples)
    return accuracy, average_train_loss