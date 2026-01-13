from pettingzoo.classic import texas_holdem_no_limit_v6
import os
print(os.environ["PATH"])

import torch
print(torch.version.cuda)

import random
from models.model import DummyModel

def predict(model, obs, legal_moves):
    state = torch.tensor(obs["observation"], dtype=torch.float32)
    probs = model(state)
    
    mask = torch.tensor(legal_moves, dtype=torch.float32)
    masked_probs = mask * probs
    masked_probs = masked_probs / masked_probs.sum()

    dist = torch.distributions.Categorical(masked_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob

def run_simulation(num_games=10):
    env = texas_holdem_no_limit_v6.env()
    env.reset()
    
    state_size = len(env.observe(env.possible_agents[0])["observation"])

if __name__ == "__main__":
    results = run_simulation()
    print("RESULTS:")
    for agent, score in results.items():
        print(agent, score)