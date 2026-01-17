import random
import torch
from pypokerengine.players import BasePokerPlayer

import sys, os
sys.path.append(os.path.dirname(__file__))

from encoder import encoder
from architecture import Poker_LSTM

lstm_out_map = {0: "call",
                1: "fold",
                2: "check",
                3: "raise small",
                4: "raise mid",
                5: "raise big",
                }

def build_action_mask(valid_actions):
    mask = [0, 0, 0, 0, 0, 0]

    for a in valid_actions:
        if a["action"] == "call":
            mask[0] = 1
        elif a["action"] == "fold":
            mask[1] = 1
        elif a["action"] == "check":
            mask[2] = 1
        elif a["action"] == "raise":
            mask[3] = 1
            mask[4] = 1
            mask[5] = 1

    return mask
    
def masked_softmax(probs, mask, eps=1e-8):
    masked = [p * m for p, m in zip(probs, mask)]
    total = sum(masked)

    if total < eps:
        # fallback: uniform over valid actions
        valid_count = sum(mask)
        return [m / valid_count for m in mask]

    return [p / total for p in masked]

class WrapperForModel(BasePokerPlayer):
    def __init__(self, model_path=None, device="cpu"):
        self.device = device

        self.model = Poker_LSTM(hidden_size=128).to(device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.game_history = {"rounds": []}
        self.hole_cards_by_action = []


    def declare_action(self, valid_actions, hole_card, round_state):
        self.hole_cards_by_action.append(hole_card)

        encoded = encoder(self.game_history, self.hole_cards_by_action)
        encoded = encoded.unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model(encoded)[0].cpu().tolist()

        mask = build_action_mask(valid_actions)
        probs = masked_softmax(probs, mask)

        action_idx = random.choices(range(len(probs)), weights=probs)[0]
        desired_action = lstm_out_map[action_idx]

        

        if desired_action == "fold":
            return "fold", 0

        if desired_action == "check":
            return "check", 0
        
        if desired_action == "call":
            call_act = next(a for a in valid_actions if a["action"] == "call")
            return "call", call_act["amount"]
        
        raise_act = next(a for a in valid_actions if a["action"] == "raise")
        min_raise = raise_act["amount"]["min"]
        max_raise = raise_act["amount"]["max"]

        if desired_action == "raise small":
            return "raise", min_raise
        
        if desired_action == "raise mid":
            return "raise", min(max_raise, 2 * min_raise)
        
        if desired_action == "raise big":
            return "raise", min(max_raise, 4 * min_raise)

        return "fold", 0

    # ---- Required callbacks (no-ops) ----
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
