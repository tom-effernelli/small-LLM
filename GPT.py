import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Opening dataset
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating encoding and decoding functions to turn characters into numbers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Splitting training and validation dataset
k = int(0.9*len(data))
train_data = data[:k]
val_data = data[k:]

batch_size = 4
block_size = 8

# Returns batchs of data blocks with size block_size
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Selecting random batches in the dataset
    x = torch.stack([data[i:i+block_size] for i in ix]) # Input
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Desired output
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits