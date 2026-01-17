import torch
import torch.nn as nn
import torch.nn.functional as F


class Poker_LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input=11,
                            hidden_size=hidden_size,
                            num_layers=2, 
                            batch_first=True)
        
        self.human_policy = nn.Linear(hidden_size, 6)
        self.alien_policy = nn.Linear(hidden_size, 6)
        #0 call
        #1 fold
        #2 check
        #3 raise small
        #4 raise mid
        #5 raise big

        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, x, mode="alien", alpha=0.5):

        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]

        human_logits = self.human_policy(features)
        alien_logits = self.alien_policy(features)

        human_probs = F.softmax(human_logits, dim=-1)
        alien_probs = F.softmax(alien_logits, dim=-1)

        if mode == "human":
            policy = human_probs
        elif mode == "alien":
            policy = alien_probs
        elif mode == "blend":
            policy = alpha * human_probs + (1 - alpha) * alien_probs
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        value = self.value_head(features).squeeze(-1)

        return policy, value