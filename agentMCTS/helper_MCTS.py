import math
import random
from typing import Optional

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action, CascadeAction
from .rules import get_legal_actions, apply_action
from .helper import copy_state, encode_state, record_state, get_all_distance_to_opponent
from .types import SeenStates
from .evaluation_play import evaluate, evaluate_new, get_f9_score_fast
from .helper_play import BoardState, detect_board_state
from .optimization import filter_meaningful_actions, order_actions, moves_next_to_stronger_opponent

C = math.sqrt(2)

# Whether the game has reached one of the termination states
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

# Node
class MCTSNode:
    def __init__(self, state, parent: Optional["MCTSNode"] = None, action=None, player_to_move_color=None):
        self.state = state              # the board state the node represents
        self.parent = parent            # the parent node
        self.action = action            # the action used to reach this node

        self.player_to_move_color = player_to_move_color           # the color of the player that is going to move next

        self.children: list[MCTSNode] = []

        self.untried_actions = None

        self.playout_num = 0
        self.reward_num = 0.0         # the number of wins for the player who made the action leading into this node

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def average_utility(self) -> float:
        if self.playout_num == 0:
            return 0.0
        return self.reward_num / self.playout_num


# {Selection}
# Calculate the UCB score for selection
def get_ucb_score(parent: MCTSNode, child: MCTSNode, exploration_constant=math.sqrt(2)) -> float:
    """
    UCB_1 = exploitation + exploration
    """

    # Defensice check
    if child.playout_num == 0:
        return float("inf")
    # Get the exploitation = average utility of the child so far
    exploitation = child.average_utility()

    # Get the exploration: Have to explore this child node enough w.r.t. its parent
    exploration = exploration_constant * math.sqrt(
        math.log(parent.playout_num) / child.playout_num
    )

    return exploitation + exploration

def select(node: MCTSNode, total_turn_count_sm: int, seen_states_sm: SeenStates) -> tuple[MCTSNode, int]:
    """
    Start from the root, apply the selection policy to choose successor states until we reach a best_selected leaf node.
    """

    # Here need to be rewrited
    while (not is_terminal(node.state, total_turn_count_sm, seen_states_sm, node.player_to_move_color)) and node.is_fully_expanded() and node.children:
        # print("DEBUG: Wrong place\n")
        # We stop at the level that is fully expanded
        # Apply selection policy to all its children and choose the best successor to see whether it is a leaf node or not
        node = max(
            node.children,
            key=lambda child: get_ucb_score(node, child)
        )
        total_turn_count_sm += 1    # Cannot be updated directly inside this func
        record_state(seen_states_sm, node.state, node.player_to_move_color)

    return node, total_turn_count_sm


# {Expansion}
def expand(node: MCTSNode, total_turn_count_sm: int, seen_states_sm: SeenStates, root_board_state: list[BoardState], my_color: PlayerColor) -> MCTSNode:
    """
    Expand a child node of the given leaf (parent) node by one untried action from this given leaf (parent) node.
    """

    # Check whether the parent has reached a terminal state--defensive--also done in the main logic
    if is_terminal(node.state, total_turn_count_sm, seen_states_sm, node.player_to_move_color):
        return node

    # Step 1: Get one of the possible untried legal actions
    # Get all its legal actions if this is the first time the node being expanded
    if node.untried_actions is None:
        node.untried_actions = get_legal_actions(
            node.state,
            node.player_to_move_color,
            total_turn_count_sm
        )

    # Defensive Check: If there is no legal action of the current node 
    if not node.untried_actions: 
        return node

    # Find the best action and apply it first
    ordered_untried_actions = order_actions(
        node.state,
        node.player_to_move_color,
        node.untried_actions,
        total_turn_count_sm,
        seen_states_sm,
        root_board_state
    )

    if node.player_to_move_color != my_color:
        k = min(3, len(ordered_untried_actions))
        action = random.choice(ordered_untried_actions[:k])
    else:
        action = ordered_untried_actions[0]

    # Remove this chosen action from the parent node
    node.untried_actions.remove(action)

    # Step 2: Apply the action and generate a new successor (child) node
    next_state = copy_state(node.state)
    apply_action(next_state, node.player_to_move_color, action)

    next_player_color = node.player_to_move_color.opponent

    child = MCTSNode(
        state=next_state,
        parent=node,
        action=action,
        player_to_move_color=next_player_color
    )

    # Step 3: Add this newly generated child to the tree--append to its parent
    node.children.append(child)

    return child


# {Simulation}
def playout(board: dict[Coord, CellState], my_color: PlayerColor, player_to_move_color: PlayerColor, total_turn_count_sm: int, seen_states_sm: SeenStates,  root_board_state: list[BoardState], rollout_depth=30) -> tuple[float, PlayerColor | None]:
    """
    Simulate a game from the current state.
    Return reward from my_color's perspective.
    """

    # Extract the current board state from the given info--the newly expanded child node's board state, player, and total turn count so far
    # Initialize the playout
    current_board = copy_state(board)
    current_player_color = player_to_move_color
    current_total_turn_count = total_turn_count_sm
    current_seen_state = seen_states_sm.copy()

    # Apply the playout policies and keep playing in turns till either:
    # 1. We have fully played the game and hit one of the terminal conditions of the game
    # 2. We have played the game till an enough depth (predefined)
    for _ in range(rollout_depth):

        # Check whether we have fully played the game
        if is_terminal(current_board, current_total_turn_count, current_seen_state, current_player_color):
            break

        # Step 1: Get all legal actions for the current player
        legal_actions = get_legal_actions(
            current_board,
            current_player_color,
            current_total_turn_count
        )
        legal_actions = filter_meaningful_actions(current_board, current_player_color, legal_actions, root_board_state)

        # Defensive check
        if not legal_actions:
            break

        # Step 2: Choose the action with the highest evaluation score
        evaluations = {}
        for action in legal_actions:
            # ss1: Apply the action
            new_board = copy_state(current_board)
            apply_action(new_board, current_player_color, action)
            penalty = 0

            if moves_next_to_stronger_opponent(new_board, current_player_color, action):
                penalty = -300

            # ss2: Evaluate the new board state after applying this action--Always with respect to our own agent player
            player_stacks = [(c, s) for c, s in current_board.items() if s.color == my_color]
            opponent_stacks = [(c, s) for c, s in current_board.items() if s.color == my_color.opponent]
            #parent_board_state = detect_board_state(current_board, opponent_stacks, player_stacks)
            #(new_board, my_color, current_total_turn_count, current_seen_state, root_board_state)
            evaluations[action] = evaluate(new_board, my_color, current_total_turn_count, current_seen_state)
            evaluations[action] += penalty

        if evaluations:
            ordered_evaluations_des = sorted(
                evaluations.items(),
                key=lambda item: item[1],
                reverse=True
            )
            ordered_evaluations_asc = sorted(
                evaluations.items(),
                key=lambda item: item[1],
                reverse=False
            )

            if current_player_color != my_color:
                k = min(3, len(ordered_evaluations_asc))
                action = random.choice(ordered_evaluations_asc[:k])[0]
            else:
                action = ordered_evaluations_des[0][0]
        else:
            action = random.choice(legal_actions)

        # Step 3: Update
        # Update the board
        apply_action(current_board, current_player_color, action)

        # Update the total turn count and the seen state dict after the current iteration
        current_total_turn_count += 1
        record_state(current_seen_state, current_board, current_player_color)

        # Switch to the other player
        current_player_color = current_player_color.opponent
        

    # Get the reward value of this playout at the end of the playout (we have terminate this playout)
    return get_score_for_playout_result(current_board, my_color, current_player_color, current_total_turn_count, current_seen_state)

def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def get_score_for_playout_result(
    current_board: dict[Coord, CellState],
    my_color: PlayerColor,
    current_player_color: PlayerColor,
    current_total_turn_count: int,
    current_seen_states: SeenStates
) -> tuple[float, PlayerColor | None]:

    player_stacks = [(c, s) for c, s in current_board.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in current_board.items() if s.color == my_color.opponent]

    my_score = sum(s.height for _, s in player_stacks)
    enemy_score = sum(s.height for _, s in opponent_stacks)

    # Real terminal result: keep this sharp.
    if is_terminal(current_board, current_total_turn_count, current_seen_states, current_player_color):
        if my_score > 0 and enemy_score == 0:
            return 1.0, my_color
        if enemy_score > 0 and my_score == 0:
            return 1.0, my_color.opponent
        if my_score > enemy_score:
            return 1.0, my_color
        if my_score < enemy_score:
            return 1.0, my_color.opponent
        return 0.5, None

    # Non-terminal rollout cutoff: soft heuristic reward.
    total_height = my_score + enemy_score
    material_score = 0.0
    if total_height > 0:
        material_score = (my_score - enemy_score) / total_height

    total_stacks = len(player_stacks) + len(opponent_stacks)
    stack_score = 0.0
    if total_stacks > 0:
        stack_score = (len(player_stacks) - len(opponent_stacks)) / total_stacks

    safe_escape_num = get_f9_score_fast(
        current_board,
        my_color,
        opponent_stacks,
        player_stacks
    )

    max_escape_num = max(1, 4 * len(opponent_stacks))
    escape_score = 1.0 - (safe_escape_num / max_escape_num)
    escape_score = 2.0 * escape_score - 1.0

    # Final-stack positions should care more about trapping.
    if len(opponent_stacks) == 1:
        soft_reward = (
            0.5
            + 0.15 * material_score
            + 0.15 * stack_score
            + 0.30 * escape_score
        )
    else:
        soft_reward = (
            0.5
            + 0.25 * material_score
            + 0.20 * stack_score
            + 0.10 * escape_score
        )

    soft_reward = clamp(soft_reward, 0.05, 0.95)

    if soft_reward > 0.52:
        return soft_reward, my_color
    if soft_reward < 0.48:
        return 1.0 - soft_reward, my_color.opponent

    return 0.5, None


# {Backpropagation}
def backpropagate(
    node: MCTSNode,
    reward: float,
    win_player_color: PlayerColor | None
) -> None:
    while node is not None:
        node.playout_num += 1

        if win_player_color is None:
            node.reward_num += 0.5
        elif node.parent is not None and node.parent.player_to_move_color == win_player_color:
            node.reward_num += reward

        node = node.parent
