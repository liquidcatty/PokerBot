from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
import random


def predict(env, agent):
    legal_moves = env.observe(agent)["action_mask"]
    legal_actions = [i for i, valid in enumerate(legal_moves) if valid]
    return legal_actions[random.randint(0, len(legal_actions) - 1)]

def run_simulation(num_games=10):
    env = texas_holdem_no_limit_v6.env()
    wins = {agent: 0 for agent in env.possible_agents}

    for _ in range(num_games):
        env.reset()

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            wins[agent] += reward

            if termination or truncation:
                env.step(None)
                continue

            action = predict(env=env, agent=agent)
            env.step(action)
                

    env.close()

    return wins

if __name__ == "__main__":
    results = run_simulation()
    print("RESULTS:")
    for agent, score in results.items():
        print(agent, score)