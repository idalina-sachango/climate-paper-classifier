import torch
import torch.nn as nn
import math
import torch.optim as optim
from processing import Processing
from torch.utils.data import Dataset, DataLoader

embed_len = 128
max_tokens = 50
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.data = Processing()
        self.data.process_all()
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.data.vocab), embedding_dim=128)
        self.conv1 = nn.Conv1d(128, 32, kernel_size=7, padding="same")
        self.linear = nn.Linear(32, 4)

    def forward(self, X_batch):
        print(X_batch)
        x = self.embedding_layer(X_batch)
        x = x.reshape(len(x), embed_len, max_tokens) ## Embedding Length needs to be treated as channel dimension

        x = F.relu(self.conv1(x))

        x, _ = x.max(dim=-1)

        x = self.linear(x)

        return x


