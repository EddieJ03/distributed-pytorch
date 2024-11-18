import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
import os


class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self):
        """Initialize the tokenizer with an empty vocabulary."""
        self.vocab = set()
        self.stoi = {'<pad>': 0, '<unk>': 1}
        self.itos = {0: '<pad>', 1: '<unk>'}
        self.vocab_size = 2  # Starting with <pad> and <unk> tokens

    def update_vocab(self, text):
        """Update vocabulary with new text."""
        tokens = word_tokenize(text)
        new_tokens = set(tokens) - self.vocab

        for token in new_tokens:
            index = self.vocab_size
            self.vocab.add(token)
            self.stoi[token] = index
            self.itos[index] = token
            self.vocab_size += 1

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = word_tokenize(text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])