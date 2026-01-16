import torch

suit_map = {"s":0, "h":1, "d":2, "c":3}
rank_map = {"2":0, "3":1, "4":2, "5":3, "6":4, "7":5, "8":6, "9":7, "T":8, "J":9, "Q":10, "K":11, "A":12}

def encode_card(card):
    rank, suit = card[0], card[1].lower()
    return rank_map[rank]*4 + suit_map[suit]

action_map = {"fold": 0, "check": 1, "call": 2, "raise": 3}

def encoder(game_history, num_players=5):
    encoded_seq = []

    for round_info in game_history["rounds"]:
        shared_cards = round_info.get("community_card", [])
        cards_encoded = [encode_card(c) for c in shared_cards]

        if len(cards_encoded) < 5:
            cards_encoded += [-1] * (5-len(cards_encoded))

        for action in round_info["actions"]:
            player_idx = action["player"]
            action_type = action["action"]
            amount = action.get("amount", 0)
            stack_before = action.get("stack_before", 0)
            step_vector = [player_idx, action_map[action_type], amount, stack_before] + cards_encoded


            encoded_seq.append(step_vector)
    return torch.tensor(encoded_seq, dtype=torch.float32)