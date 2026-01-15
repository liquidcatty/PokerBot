import sys
import os

# Adds the parent directory (PokerBot) to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import DummyModel
from pettingzoo.classic import texas_holdem_no_limit_v6

import torch

def predict(model, obs, legal_moves):
    state = torch.tensor(obs["observation"], dtype=torch.float32)
    probs = model(state)
    
    mask = torch.tensor(legal_moves, dtype=torch.float32)
    masked_probs = probs * mask
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    dist = torch.distributions.Categorical(masked_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob


def run_one_unit_of_training(num_games=10):
    env = texas_holdem_no_limit_v6.env()
    env.reset()
    
    state_size = len(env.observe(env.possible_agents[0])["observation"])
    action_size = len(env.observe(env.possible_agents[0])["action_mask"])
    
    model = DummyModel(state_size, 32, action_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for game in range(num_games):
        print(f"Game: {game+1}/{num_games}")
        env.reset()
        log_probs = {agent: [] for agent in env.possible_agents}
        rewards = {agent: [] for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            rewards[agent].append(reward)

            if termination or truncation:
                env.step(None)
                continue

            legal_moves = obs["action_mask"]
            action, log_prob = predict(model, obs, legal_moves)
            log_probs[agent].append(log_prob)

            env.step(action)

        for agent in env.possible_agents:
            torch.autograd.set_detect_anomaly(True)

            disc_rewards = []
            R = 0
            
            for r in reversed(rewards[agent]):
                R = r + R * 0.99
                disc_rewards.insert(0, R)
            
            disc_rewards = torch.tensor(disc_rewards)
            disc_rewards = ((disc_rewards - disc_rewards.mean())/(disc_rewards.std() + 0.000001)).detach()

            loss = 0.0
            for log_prob, G in zip(log_probs[agent], disc_rewards):
                loss -= log_prob * G

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    env.close()
    print(rewards)
    return rewards


if __name__ == "__main__":
    results = run_one_unit_of_training()
    print("RESULTS:")
    for agent, score in results.items():
        print(agent, score)