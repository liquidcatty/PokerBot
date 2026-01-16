import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input=11, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        probs = F.softmax(logits, dim=-1)
        return probs