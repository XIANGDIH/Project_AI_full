# This file contains the logic about the game play-strategy for the play phase.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action
from .rules import get_legal_actions
from .evaluation_play import evaluate, evaluate_new
from .rules import apply_action
from .types import SeenStates
from .helper import encode_state, record_state


# ----------------------------
# Implementation of MINIMAX
# ----------------------------

def choose_action(board: dict[Coord, CellState], my_color: PlayerColor, depth: int, total_turn_count: int, seen_states: SeenStates) -> Action:
    score, best_action = minimax(
        board=board,
        depth=depth,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing=True,
        my_color=my_color,
        total_turn_count=total_turn_count,
        seen_states=seen_states
    )
    return best_action

def minimax(board: dict[Coord, CellState], depth: int, alpha: float, beta: float, maximizing: bool, my_color: PlayerColor, total_turn_count: int, seen_states: SeenStates) -> tuple[int, Action]:
    """
    Using DFS to implement the MINIMAX strategy with alpha-beta pruning as cut-offs
    Returns (score, best_action)
    """

    current_color = my_color if maximizing else my_color.opponent

    # Base case
    if depth == 0 or is_terminal(board, total_turn_count, seen_states, current_color):
        return evaluate(board, my_color), None

    # Decide whose turn is it and get all legal actions (a list of actions) for this turn
    legal_actions = get_legal_actions(board, current_color, total_turn_count)

    # Defensive check
    if not legal_actions:
        return evaluate(board, my_color), None

    # The action we are going to return for this tree
    best_action = None

    # Our turn--Maximizing part
    if maximizing:
        # Start with the worst 
        best_score = float("-inf")

        for action in legal_actions:
            # Step 1: Generate the successor of the specific legal action
            # Since we are on the MAX level, the successor should be on the MIN level below
            next_state = copy_state(board)
            apply_action(next_state, my_color, action)

            # Step 2: Perform minimax on this new successor
            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                False,
                my_color,
                total_turn_count,
                seen_states
            )

            # Step 3: Check whether the new evaluation value gives a better score, update it if it gives
            if score > best_score:
                best_score = score
                best_action = action

            # Step 4: Update the best score the current MAX level that could already be guaranteed
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break   # beta cut-off

        # Return the best score and the corresponding best action for this node
        return best_score, best_action

    else:
        best_score = float("inf")
        opponent = my_color.opponent

        for action in legal_actions:
            next_state = copy_state(board)
            apply_action(next_state, opponent, action)

            score, _ = minimax(
                next_state,
                depth - 1,
                alpha,
                beta,
                True,
                my_color,
                total_turn_count,
                seen_states
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