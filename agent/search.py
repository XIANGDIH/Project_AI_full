# This file contains the logic about the game play-strategy for the play phase.


import random
import time

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action, CascadeAction
from .rules import get_legal_actions
from .evaluation_play import evaluate, evaluate_new, get_f8_score
from .rules import apply_action
from .types import SeenStates
from .helper import encode_state, record_state, copy_state
from .helper_play import BoardState, detect_board_state
from .optimization import is_meaningless_cascade, order_actions, moves_next_to_stronger_opponent, move_allows_direct_cascade_elimination


# ----------------------------
# Implementation of MINIMAX
# ----------------------------

class SearchTimeout(Exception):
    pass


def choose_emergency_action(board, my_color, total_turn_count, seen_states, root_board_state=None):
    legal_actions = get_legal_actions(board, my_color, total_turn_count)
    if not legal_actions:
        return None, float("-inf")

    player_stacks = [(c, s) for c, s in board.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]
    if root_board_state is None:
        root_board_state = detect_board_state(board, opponent_stacks, player_stacks)

    origin_opponent_num = len(opponent_stacks)
    best_action = random.choice(legal_actions)
    best_score = float("-inf")

    try:
        candidate_actions = order_actions(board, my_color, legal_actions, total_turn_count, seen_states, root_board_state)
    except Exception:
        candidate_actions = legal_actions

    for action in candidate_actions:
        next_state = copy_state(board)
        apply_action(next_state, my_color, action)

        new_opponent_num = sum(
            1 for cell in next_state.values()
            if cell.color == my_color.opponent
        )

        if new_opponent_num == 0:
            return action, 1000000.0

        score = evaluate_new(next_state, my_color, total_turn_count + 1, seen_states, root_board_state)
        score += get_f8_score(next_state, seen_states)

        if new_opponent_num < origin_opponent_num:
            score += 500

        if moves_next_to_stronger_opponent(next_state, my_color, action):
            score -= 300
        if move_allows_direct_cascade_elimination(next_state, my_color, action):
            score -= 300

        if score > best_score:
            best_score = score
            best_action = action

    return best_action, best_score


def choose_action(board, my_color, max_depth, total_turn_count, seen_states, time_limit=None):

    player_stacks = [(c, s) for c, s in board.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]
    root_board_state = detect_board_state(board, opponent_stacks, player_stacks)
    best_action, best_score = choose_emergency_action(board, my_color, total_turn_count, seen_states, root_board_state)
    if best_score >= 1000000.0:
        return best_action, best_score

    deadline = None
    if time_limit is not None:
        deadline = time.perf_counter() + time_limit

    for depth in range(1, max_depth + 1):
        try:
            score, action = minimax(
                board=board,
                depth=depth,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=True,
                my_color=my_color,
                total_turn_count=total_turn_count,
                seen_states=seen_states,
                root_board_state=root_board_state,
                deadline=deadline
            )
        except SearchTimeout:
            break

        if action is not None:
            best_action = action
            best_score = score

    if best_action is None:
        legal_actions = get_legal_actions(board, my_color, total_turn_count)
        return legal_actions[0], float("-inf")

    return (best_action, best_score)

def minimax(board: dict[Coord, CellState], depth: int, alpha: float, beta: float, maximizing: bool, my_color: PlayerColor, total_turn_count: int, seen_states: SeenStates, root_board_state: list[BoardState], deadline=None) -> tuple[int, Action]:
    """
    Using DFS to implement the MINIMAX strategy with alpha-beta pruning as cut-offs
    Returns (score, best_action)
    """

    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout

    # Get who is playing in this turn--the new board is obtained by which side's action
    current_color = my_color if maximizing else my_color.opponent

    # Don't wanna change the counter and seen state container in the game
    total_turn_count_mm = total_turn_count

    # Base case
    if depth == 0 or is_terminal(board, total_turn_count, seen_states, current_color):
        return evaluate_new(board, my_color, total_turn_count, seen_states, root_board_state), None

    # Decide whose turn is it and get all legal actions (a list of actions) for this turn
    legal_actions = get_legal_actions(board, current_color, total_turn_count)

    # Defensive check
    if not legal_actions:
        return evaluate_new(board, my_color, total_turn_count, seen_states, root_board_state), None

    # The action we are going to return for this tree
    best_action = None

    # Our turn--Maximizing part
    if maximizing:
        # Start with the worst 
        best_score = float("-inf")

        for action in order_actions(board, my_color, legal_actions, total_turn_count, seen_states, root_board_state):
            # Dealing with the meaningless case corresponding to feature 1 in our evaluation function
            # Check whether the current action is meaningful
            if is_meaningless_cascade(board, my_color, action):
                continue

            # Step 1: Generate the successor of the specific legal action
            # Since we are on the MAX level, the successor should be on the MIN level below
            next_state = copy_state(board)
            apply_action(next_state, my_color, action)

            # Get the penalty score
            penalty = get_f8_score(next_state, seen_states)

            # Add this new successor to the seen state dictionary--not sure whether record it here
            branch_seen_states = seen_states.copy()
            # After my agent moves, the opponent is about to move
            next_color = my_color.opponent
            record_state(branch_seen_states, next_state, next_color)

            # Step 2: Perform minimax on this new successor
            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                False,
                my_color,
                total_turn_count_mm + 1,
                branch_seen_states,
                root_board_state,
                deadline
            )

            score += penalty
            # Check whether the current action is not preferred
            if moves_next_to_stronger_opponent(next_state, my_color, action):
                score -= 300
            if move_allows_direct_cascade_elimination(next_state, my_color, action):
                score -= 300

            # Step 3: Check whether the new evaluation value gives a better score, update it if it gives
            if score > best_score:
                best_score = score
                best_action = action

            # Step 4: Update the best score the current MAX level that could already be guaranteed
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break   # beta cut-off

        if best_action is None:
            return evaluate_new(board, my_color, total_turn_count, seen_states, root_board_state), None

        # Return the best score and the corresponding best action for this node
        return best_score, best_action

    else:
        best_score = float("inf")
        opponent_color = my_color.opponent

        for action in order_actions(board, opponent_color, legal_actions, total_turn_count, seen_states, root_board_state):
            if is_meaningless_cascade(board, opponent_color, action):
                continue

            next_state = copy_state(board)
            apply_action(next_state, opponent_color, action)

            penalty = get_f8_score(next_state, seen_states)

            branch_seen_states = seen_states.copy()
            # After the opponent agent moves, my agent is about to move
            next_color = opponent_color.opponent
            record_state(branch_seen_states, next_state, next_color)

            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                True,
                my_color,
                total_turn_count_mm + 1,
                branch_seen_states,
                root_board_state,
                deadline
            )

            score += penalty
            if moves_next_to_stronger_opponent(next_state, opponent_color, action):
                score += 300
            if move_allows_direct_cascade_elimination(next_state, opponent_color, action):
                score += 300

            if score < best_score:
                best_score = score
                best_action = action

            # Update the best score the current MIN level that could already be guaranteed
            beta = min(beta, best_score)

            if beta <= alpha:
                break   # alpha cut-off

        if best_action is None:
            return evaluate_new(board, my_color, total_turn_count, seen_states, root_board_state), None

        return best_score, best_action
    

# Directly related helper functions

def is_terminal(board: dict[Coord, CellState], total_turn_count: int, seen_states: SeenStates, color: PlayerColor) -> bool:
    # Termination condition 1: All of a player's tokens are removed
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]
    if len(blue_stacks) == 0 or len(red_stacks) == 0:
        return True
    
    # Termination condition 2: The play phase has ran 300 turns
    if total_turn_count - 8 >= 300:
        return True
    
    # Termination condition 3: The same board position occurs three times
    encoded_state = encode_state(board)
    if encoded_state in seen_states:
        seen_count, seen_color = seen_states[encoded_state]

        # If this same player is about to see the same state for the third time
        if seen_count >= 2 and seen_color == color:
            return True
    
    return False
    
