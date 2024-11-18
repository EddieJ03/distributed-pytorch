### ENCODER ###

import torch
import torch.nn as nn
from torch.nn import functional as F
from TrainingConstants import n_embd, block_size, feed_forward, n_head, n_hidden, n_layer, n_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.3

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, decoding=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.decoding = decoding

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5

        if self.decoding:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)

        attention_maps.append(wei)

        # wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, decoding=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, decoding) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps, dropout=False):
        out = torch.cat([h(x, attention_maps) for h in self.heads], dim=-1)

        if dropout:
            return self.dropout(self.proj(out))

        return self.proj(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, feed_forward),
            nn.ReLU(),
            nn.Linear(feed_forward, n_embd),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dropout=False):
        if dropout:
            return self.dropout(self.net(x))

        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head=n_head, decoding=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size, decoding)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_maps=None, dropout=False):
        x = x + self.sa(self.ln1(x), attention_maps, dropout)

        x = x + self.ffwd(self.ln2(x), dropout)

        return x

class Classifier(nn.Module):
    def __init__(self, vocab_size, input_size=n_embd, hidden_size=n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer.
        self.fc2 = nn.Linear(hidden_size, n_output)  # Second fully connected layer, outputting three classes.
        self.encoder = Encoder(vocab_size, n_head, n_layer)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x, attn_maps = self.encoder(x)
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer.
        x = self.fc2(x)  # Pass the result to the second layer.
        return x, attn_maps


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_head=n_head, n_layer=n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, decoding=False) for _ in range(n_layer)])

    def forward(self, idx):
        tok_emb = self.token_embedding_table(idx)

        # absolute positional encoding
        # div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))

        # pos = torch.arange(block_size, dtype=torch.float).reshape(block_size, 1)

        # stacked = torch.stack([torch.sin(pos * div_term), torch.cos(pos * div_term)], dim=2)

        # stacked = stacked.to(device)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pos_emb = self.position_embedding_table(torch.arange(block_size, device=device))

        # stacked = torch.stack([pos_emb, pos_emb], dim=2)

        tok_emb = tok_emb.to(device)

        pos_emb = pos_emb.to(device)

        # x = tok_emb + torch.flatten(stacked, start_dim=1, end_dim=2)

        x = tok_emb + pos_emb

        attention_maps = []

        for block in self.blocks:
           x = block(x, attention_maps, True)

        x = torch.mean(x, dim=1)

        return x, attention_maps