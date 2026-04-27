# This file contains the logic about the game play-strategy for the play phase

# ----------------------------
# Implementation of MINIMAX
# ----------------------------

def minimax(state, depth, is_maximizing_player):
    if depth == 0 or is_terminal(state):
        return evaluate(state), None

    best_action = None

    if is_maximizing_player:
        max_eval = float('-inf')

        for action in get_legal_actions(state):
            new_state = apply_action(state, action)
            eval, _ = minimax(new_state, depth - 1, False)

            if eval > max_eval:
                max_eval = eval
                best_action = action

        return max_eval, best_action

    else:
        min_eval = float('inf')

        for action in get_legal_actions(state):
            new_state = apply_action(state, action)
            eval, _ = minimax(new_state, depth - 1, True)

            if eval < min_eval:
                min_eval = eval
                best_action = action

        return min_eval, best_action