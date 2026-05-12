# This file contains the logic about the game play-strategy for the play phase.


import os

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action, CascadeAction
from .rules import get_legal_actions
from .evaluation_play import evaluate, evaluate_new
from .rules import apply_action
from .types import SeenStates
from .helper import encode_state, record_state, meaningful_cascade


# ----------------------------
# Implementation of MINIMAX
# ----------------------------

def decision_trace_enabled() -> bool:
    text = os.getenv("AGENT_DECISION_TRACE", "0")
    return text.strip().lower() in ["1", "true", "yes", "on"]

def action_to_text(action: Action | None) -> str:
    if action is None:
        return "None"

    action_text = str(action)
    action_text = action_text.replace("PlaceAction", "PLACE")
    action_text = action_text.replace("MoveAction", "MOVE")
    action_text = action_text.replace("EatAction", "EAT")
    action_text = action_text.replace("CascadeAction", "CASCADE")
    return action_text


def choose_action(board: dict[Coord, CellState], my_color: PlayerColor, depth: int, total_turn_count: int, seen_states: SeenStates) -> Action:
    score, best_action = minimax(
        board=board,
        depth=depth,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing=True,
        my_color=my_color,
        total_turn_count=total_turn_count,
        seen_states=seen_states,
        root_depth=depth
    )
    if decision_trace_enabled():
        print(f"DecisionTrace: {my_color} chooses {action_to_text(best_action)} with score {score}")
    return best_action

def minimax(
    board: dict[Coord, CellState],
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    my_color: PlayerColor,
    total_turn_count: int,
    seen_states: SeenStates,
    root_depth: int
) -> tuple[int, Action]:
    """
    Using DFS to implement the MINIMAX strategy with alpha-beta pruning as cut-offs
    Returns (score, best_action)
    """

    current_color = my_color if maximizing else my_color.opponent
    # Don't wanna change the counter and seen state container in the game
    total_turn_count_mm = total_turn_count
    seen_states_mm = seen_states.copy()

    # Base case
    if depth == 0 or is_terminal(board, total_turn_count, seen_states, current_color):
        return evaluate_new(board, my_color, total_turn_count), None

    # Decide whose turn is it and get all legal actions (a list of actions) for this turn
    legal_actions = get_legal_actions(board, current_color, total_turn_count)

    # Defensive check
    if not legal_actions:
        return evaluate_new(board, my_color, total_turn_count), None

    # The action we are going to return for this tree
    best_action = None

    # Our turn--Maximizing part
    if maximizing:
        # Start with the worst 
        best_score = float("-inf")

        for action in legal_actions:
            # Dealing with the meaningless case corresponding to feature 1 in our evaluation function
            # Check whether the current action is meaningful
            if is_meaningless_cascade(board, my_color, action):
                continue

            # Step 1: Generate the successor of the specific legal action
            # Since we are on the MAX level, the successor should be on the MIN level below
            next_state = copy_state(board)
            apply_action(next_state, my_color, action)

            # Add this new successor to the seen state dictionary--not sure whether record it here
            record_state(seen_states_mm, next_state, my_color)

            # Step 2: Perform minimax on this new successor
            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                False,
                my_color,
                total_turn_count_mm + 1,
                seen_states_mm,
                root_depth
            )

            if decision_trace_enabled() and depth == root_depth:
                print(
                    f"DecisionTrace: candidate {action_to_text(action)} -> score {score} "
                    f"(alpha={alpha}, beta={beta})"
                )

            # Step 3: Check whether the new evaluation value gives a better score, update it if it gives
            if score > best_score:
                best_score = score
                best_action = action

            # Step 4: Update the best score the current MAX level that could already be guaranteed
            alpha = max(alpha, best_score)

            if beta <= alpha:
                if decision_trace_enabled() and depth == root_depth:
                    print(
                        f"DecisionTrace: prune at root after {action_to_text(action)} "
                        f"(alpha={alpha}, beta={beta})"
                    )
                break   # beta cut-off

        # Return the best score and the corresponding best action for this node
        return best_score, best_action

    else:
        best_score = float("inf")
        opponent_color = my_color.opponent

        for action in legal_actions:
            

            if is_meaningless_cascade(board, opponent_color, action):
                continue

            next_state = copy_state(board)
            apply_action(next_state, opponent_color, action)

            record_state(seen_states_mm, next_state, my_color)

            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                True,
                my_color,
                total_turn_count_mm + 1,
                seen_states,
                root_depth
            )

            if score < best_score:
                best_score = score
                best_action = action

            # Update the best score the current MIN level that could already be guaranteed
            beta = min(beta, best_score)

            if beta <= alpha:
                break   # alpha cut-off

        return best_score, best_action
    

def copy_state(state):
    return state.copy()

def is_terminal(board: dict[Coord, CellState], total_turn_count: int, seen_states: SeenStates, color: PlayerColor) -> bool:
    # Termination condition 1: All of a player's tokens are removed
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]
    if len(blue_stacks) == 0 or len(red_stacks) == 0:
        return True
    
    # Termination condition 2: The play phase has ran 300 turns
    if total_turn_count + 1 - 4 >= 300:
        return True
    
    # Termination condition 3: The same board position occurs three times
    encoded_state = encode_state(board)
    if encoded_state in seen_states:
        seen_count, seen_color = seen_states[encoded_state]

        # If this same player is about to see the same state for the third time
        if seen_count >= 2 and seen_color == color:
            return True
    
    return False

def is_meaningless_cascade (new_copied_state: dict[Coord, CellState], my_color: PlayerColor, action_to_be_applied: Action) -> bool:
    player_stacks = [(c, s) for c, s in new_copied_state.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in new_copied_state.items() if s.color == my_color.opponent]
    # If it's other actions
    if not isinstance(action_to_be_applied, CascadeAction):
        return False
    
    # Check whether the new cascade action is meaningful
    attacker_coord = action_to_be_applied.coord
    attacker_state = CellState(my_color, new_copied_state[attacker_coord].height)
    attacking_direction = action_to_be_applied.direction

    is_meaningful = meaningful_cascade(attacker_coord, attacker_state, opponent_stacks, attacking_direction)

    if is_meaningful:
        #print("DEBUG: Here--meaningful wrong\n")
        return False
    
    return True
    
