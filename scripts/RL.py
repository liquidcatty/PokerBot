from pypokerengine.api.game import setup_config, start_poker
import random
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.model import DummyAlgorithm



def run_singular_poker_tournament(num_players=5, max_rounds=20):
    config = setup_config(max_round=max_rounds,
                          initial_stack=1000,
                          small_blind_amount=10)
    for i in range(num_players):
        config.register_player(name=f"p{i}", algorithm=DummyAlgorithm())

    game_result = start_poker(config=config,
                              verbose=0)
    return game_result

def run_one_set_of_games(num_players=5, num_games=10):
    results=[]
    for game in range(num_games):
        game_results = []
        
        print(f"game: {game+1}/{num_games}")
        max_rounds = random.randint(10, 30)
        result = run_singular_poker_tournament(max_rounds=max_rounds)
        
        players_info = result['players']
        for player_info in players_info:
            game_results.append(player_info["stack"])

        results.append(game_results)

    return results
        
def training_epoch():
    pass

def training(num_epochs=10, num_players=5, num_games_per_epoch=10):
    pass
        

print(run_one_set_of_games())