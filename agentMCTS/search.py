# This file contains the logic about the game play-strategy for the play phase.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action
from .rules import get_legal_actions
from .evaluation_play import evaluate
from .rules import apply_action
from .types import SeenStates
from .helper import encode_state, record_state
from .helper_MCTS import MCTSNode, copy_state, select, expand, is_terminal, playout, backpropagate


# ----------------------------
# Implementation of MCTS
# ----------------------------

def mcts_choose_action(
    board: dict[Coord, CellState],
    player_to_move_color: PlayerColor,
    total_turn_count: int,
    seen_states: SeenStates,
    weights: dict[str, float],
    iterations=500
) -> Action:
    # The count we take in would be the total count of the game so far (me + opponent)
    # We also take in the dictionary of all board states have been witnessed (with the number of times it has been witnessed) in the game so far

    # Initialize the tree with root node as the current board
    root = MCTSNode(
        state=copy_state(board),
        parent=None,
        action=None,
        player_to_move_color=player_to_move_color
    )
    root.untried_actions = get_legal_actions(board, player_to_move_color, total_turn_count)
    
    # Get all possible legal actions that we are going to choose the best one to return among--inside of the initialization

    # Till we run out of the time
    for _ in range(iterations):
        # Start from the node each time!!!
        node = root
        total_turn_count_sm = total_turn_count
        seen_states_sm = seen_states.copy()

        # P1: Selection
        # The node now is the parent node that is a leaf node, has the greatest UCB score, and is about to be expanded
        node = select(node, total_turn_count_sm, seen_states_sm)

        # P2: Expansion
        if not is_terminal(node.state, total_turn_count_sm, seen_states_sm, node.player_to_move_color):
            # For the root
            if node.untried_actions is None:
                node.untried_actions = get_legal_actions(
                    node.state,
                    node.player_to_move_color,
                    total_turn_count_sm
                )

            # The node now is the newly expanded child node of the previous parent node obtained by applying an intried legal function of the previous parent node
            if node.untried_actions:
                node = expand(node, player_to_move_color, total_turn_count_sm, seen_states_sm, weights)
                total_turn_count_sm += 1
                record_state(seen_states_sm, node.state, node.player_to_move_color)

        # P3: Simulation
        reward = playout(node.state, root.player_to_move_color, node.player_to_move_color, total_turn_count_sm, seen_states_sm, weights)

        # P4: Backpropagation
        backpropagate(node, reward, node.player_to_move_color)

    # Choose and return the legal action that has the greatest average utility
    return max(root.children, key=lambda child: child.average_utility()).action
