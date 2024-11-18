from Tokenizer import SimpleTokenizer
from Training import load_texts, collate_batch, train_epoch, compute_classifier_accuracy
from PresidentsDataset import SpeechesClassificationDataset

import torch
from torch.utils.data import DataLoader
import os
import pickle

from TrainingConstants import batch_size
from Encoder import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data and creating tokenizer ...")

tokenizer = SimpleTokenizer() # create a tokenizer from the data

tokenizer_filename = './tokenizer.pkl'

if os.path.exists(tokenizer_filename):
    with open(tokenizer_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    print('loaded from file')

model_path = './checkpoint.pt'

test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "./test.tsv")
test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

classifier_model = Classifier(tokenizer.vocab_size)

if os.path.exists(model_path):
      # Load the state dictionary from the file
      state_dict = torch.load(model_path, map_location=torch.device(device))  # Change to 'cuda' if using GPU

      # Load the state dictionary into the model
      classifier_model.load_state_dict(state_dict)

classifier_model.to(device)

print(f'Test Accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader, device):.3f}')
