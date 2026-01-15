from pypokerengine.api.game import setup_config, start_poker
import random
from pypokerengine.players import BasePokerPlayer


class DummyAlgorithm(BasePokerPlayer):
    """
    Simple baseline poker agent:
    - Never crashes
    - Always returns a legal action
    - Slightly prefers call over fold
    """

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        valid_actions example:
        [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 20},
            {'action': 'raise', 'amount': {'min': 40, 'max': 200}}
        ]
        """

        # Prefer non-fold actions if possible
        call_action = next((a for a in valid_actions if a["action"] == "call"), None)
        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)

        # Random but stable policy
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

def run_poker_tournament(num_players=6, max_rounds=20):
    config = setup_config(max_round=max_rounds,
                          initial_stack=1000,
                          small_blind_amount=10)
    for i in range(num_players):
        config.register_player(name=f"p{i}", algorithm=DummyAlgorithm())

    game_result = start_poker(config=config,
                              verbose=1)
    return game_result