import torch.nn as nn
import torch.nn.init as init


class FineTuningModel(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(FineTuningModel, self).__init__()
        # Adapter layer
        self.adapter = nn.Linear(embedding_dim, embedding_dim)
        init.xavier_uniform_(self.adapter.weight)
        self.adapter.bias.data.fill_(0.01)

        self.dropout = nn.Dropout(0.2)
        # Additional layers for prediction
        self.fc1 = nn.Linear(embedding_dim, 64)
        init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(64, output_dim)
        init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        # self.bn1 = nn.BatchNorm1d(int(embedding_dim / 2))
        # self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.adapter(x)
        # x = self.bn1(x)
        x = self.fc1(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
