import pickle
import torch
from Tokenizer import SimpleTokenizer
from Training import load_texts, collate_batch, train_epoch, compute_classifier_accuracy
from PresidentsDataset import SpeechesClassificationDataset
from torch.utils.data import DataLoader
from Encoder import Classifier
import os
from TrainingConstants import batch_size, epochs_CLS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data and creating tokenizer ...")

tokenizer = SimpleTokenizer() # create a tokenizer from the data

tokenizer_filename = './tokenizer.pkl'

if os.path.exists(tokenizer_filename):
    with open(tokenizer_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    print('loaded from file')
else:
    for text in load_texts('./'):
      tokenizer.update_vocab(text.split('\t', 1)[1])

    with open(tokenizer_filename, 'wb') as file:
        pickle.dump(tokenizer, file)

    print('created and stored')

print("Vocabulary size is", tokenizer.vocab_size)

print("vocab size: ", tokenizer.vocab)

import pickle

# ------------------------------Classifier Code---------------------------------- #
def run_classifier():
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "./train.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "./test.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    classifier_model = Classifier(tokenizer.vocab_size)

    total_params = sum(p.numel() for p in classifier_model.parameters())

    print("Total number of parameters:", total_params)

    model_path = './classifier_model_dict.pth'

    # have an existing file, load it in!
    if os.path.exists(model_path):
      # Load the state dictionary from the file
      state_dict = torch.load(model_path, map_location=torch.device(device))  # Change to 'cuda' if using GPU

      # Load the state dictionary into the model
      classifier_model.load_state_dict(state_dict)

    classifier_model.to(device)

    # Adam optimizer
    optimizer = torch.optim.Adam(
        classifier_model.parameters(),  
        lr=0.0001,            # You can set your desired learning rate here
        betas=(0.9, 0.98),
        eps=1e-6
    )

    for epoch in range(epochs_CLS):
        train_accuracy, train_loss = train_epoch(train_CLS_loader, classifier_model, optimizer)

        # save each epoch
        torch.save(classifier_model.state_dict(), model_path)

        print(f'Epoch #{epoch+1}: \t train accuracy {train_accuracy:.3f}\t train loss {train_loss:.3f}\t test accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader):.3f}')

# ------------------------------Classifier Code---------------------------------- #

run_classifier()