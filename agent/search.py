# This file contains the logic about the game play-strategy for the play phase

from referee.game import PlayerColor, Coord, CellState, Action, MoveAction, EatAction, CascadeAction

from .rules import get_legal_actions, apply_action
from .evaluation_play import evaluate_play_phase

# ----------------------------
# Minimal runnable search for play phase
# ----------------------------
# We first run a one-step look-ahead:
# 1) enumerate legal actions
# 2) simulate each action
# 3) evaluate the board by play-phase heuristic
# 4) choose the action with maximum score

def get_action_priority(action: Action) -> int:
    """
    Tie-break preference when two actions have same evaluation score:
    EAT > CASCADE > MOVE
    """
    if isinstance(action, EatAction):
        return 0
    if isinstance(action, CascadeAction):
        return 1
    if isinstance(action, MoveAction):
        return 2
    return 3


def choose_best_action_play_phase(
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int
) -> Action:
    legal_actions = get_legal_actions(board, color, total_turn_count)
    if len(legal_actions) == 0:
        raise ValueError("No legal action available in play phase")

    best_action = legal_actions[0]
    best_score = float("-inf")
    best_priority = get_action_priority(best_action)

    for action in legal_actions:
        # Shallow copy is enough: apply_action only replaces/removes dict entries
        next_board = dict(board)
        apply_action(next_board, color, action, verbose=False)

        score = evaluate_play_phase(next_board, color)
        priority = get_action_priority(action)

        # Max score first; then action-priority tie-break
        if (score > best_score) or (score == best_score and priority < best_priority):
            best_score = score
            best_action = action
            best_priority = priority

    return best_action
