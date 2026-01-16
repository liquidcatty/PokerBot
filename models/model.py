import random
from pypokerengine.players import BasePokerPlayer

import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from encoder import encoder
from architecture import LSTM_model

class DummyAlgorithm(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        call_action = next((a for a in valid_actions if a["action"] == "call"), None)
        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)

        r = random.random()

        if raise_action and r > 0.85:
            amount = random.randint(
                raise_action["amount"]["min"],
                raise_action["amount"]["max"]
            )
            return "raise", amount

        if call_action:
            return "call", call_action["amount"]

        return "fold", 0

    # ---- Required callbacks (no-ops) ----
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
