import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        self.fc_seq = nn.Linear(state_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x[:, -1, :]
        x = F.relu(self.fc_seq(x))
        x = self.fc_out(x)
        return F.softmax(x, dim=1)
    