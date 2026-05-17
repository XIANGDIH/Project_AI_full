# This file contains the logic about the game play-strategy for the play phase.

import random
import time

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action
from .rules import get_legal_actions
from .evaluation_play import evaluate
from .rules import apply_action
from .types import SeenStates
from .helper import copy_state, encode_state, record_state
from .helper_MCTS import MCTSNode, select, expand, is_terminal, playout, backpropagate
from .helper_play import detect_board_state, BoardState
from .optimization import filter_meaningful_actions, order_actions, moves_next_to_stronger_opponent


# ----------------------------
# Implementation of MCTS
# ----------------------------

def choose_emergency_action(
    board: dict[Coord, CellState],
    my_color: PlayerColor,
    total_turn_count: int,
    seen_states: SeenStates,
    root_board_state: list[BoardState] | None = None
) -> Action | None:
    legal_actions = get_legal_actions(board, my_color, total_turn_count)
    if not legal_actions:
        return None

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
        next_board = copy_state(board)
        apply_action(next_board, my_color, action)
        new_opponent_num = sum(
            1 for cell in next_board.values()
            if cell.color == my_color.opponent
        )

        if new_opponent_num == 0:
            return action

        score = evaluate(next_board, my_color, total_turn_count + 1, seen_states)
        if new_opponent_num < origin_opponent_num:
            score += 500
        if moves_next_to_stronger_opponent(next_board, my_color, action):
            score -= 300

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def mcts_choose_action(board: dict[Coord, CellState], player_to_move_color: PlayerColor, total_turn_count: int, seen_states: SeenStates, iterations=500, time_limit=None) -> Action:
    # The count we take in would be the total count of the game so far (me + opponent)
    # We also take in the dictionary of all board states have been witnessed (with the number of times it has been witnessed) in the game so far

    # Need to keep track of our player agent's color since evaluations are with respect to our agent player
    # Detect the board state of the root (given board) so as to adjust weight in the EVAL
    my_color = player_to_move_color
    player_stacks = [(c, s) for c, s in board.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]
    # Detect the board state of the root for weight adjustment
    root_board_state = detect_board_state(board, opponent_stacks, player_stacks)
    emergency_action = choose_emergency_action(
        board,
        my_color,
        total_turn_count,
        seen_states,
        root_board_state
    )
    legal_root_actions = filter_meaningful_actions(
        board,
        my_color,
        get_legal_actions(board, my_color, total_turn_count),
        root_board_state
    )
    if not legal_root_actions:
        return emergency_action

    # Return the action that can eliminate the opponent stack right away
    origin_opponent_num = len(opponent_stacks)
    capture_actions = []
    for action in legal_root_actions:
        next_board = copy_state(board)
        apply_action(next_board, my_color, action)
        new_opponent_num = sum(
            1 for cell in next_board.values()
            if cell.color == my_color.opponent
        )

        if new_opponent_num == 0:
            return action

        if new_opponent_num < origin_opponent_num:
            capture_actions.append(action)

        if capture_actions:
            return capture_actions[0]

    # Initialize the tree with root node as the current board
    root = MCTSNode(
        state=copy_state(board),
        parent=None,
        action=None,
        player_to_move_color=my_color
    )
    root.untried_actions = legal_root_actions
    
    # Get all possible legal actions that we are going to choose the best one to return among--inside of the initialization

    # Till we run out of the time
    deadline = None
    if time_limit is not None:
        deadline = time.perf_counter() + time_limit

    for _ in range(iterations):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        # Start from the node each time!!!
        node = root
        total_turn_count_sm = total_turn_count
        seen_states_sm = seen_states.copy()

        # P1: Selection
        # The node now is the parent node that is a leaf node, has the greatest UCB score, and is about to be expanded
        node, total_turn_count_sm = select(node, total_turn_count_sm, seen_states_sm)

        # P2: Expansion
        if not is_terminal(node.state, total_turn_count_sm, seen_states_sm, node.player_to_move_color):
            # The node now is the newly expanded child node of the previous parent node obtained by applying an intried legal function of the previous parent node
            if node.untried_actions is None or node.untried_actions:
                expanded_node = expand(node, total_turn_count_sm, seen_states_sm, root_board_state, my_color)
                if expanded_node is not node:
                    node = expanded_node
                    total_turn_count_sm += 1
                    record_state(seen_states_sm, node.state, node.player_to_move_color)

        # P3: Simulation
        reward, winner_player_color = playout(node.state, my_color, node.player_to_move_color, total_turn_count_sm, seen_states_sm, root_board_state)

        # P4: Backpropagation
        backpropagate(node, reward, winner_player_color)

    # Choose and return the legal action that has the greatest average utility
    if root.children:
        return max(root.children, key=lambda child: child.average_utility()).action

    return emergency_action
