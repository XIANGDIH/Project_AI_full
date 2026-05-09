import math
import random
from typing import Optional

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action, CascadeAction
from .rules import get_legal_actions, apply_action
from .helper import encode_state, record_state, meaningful_cascade
from .types import SeenStates
from .evaluation_play import evaluate, evaluate_new

C = math.sqrt(2)

# Whether the game has reached one of the termination states
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

def copy_state(state):
    return state.copy()

# Node
class MCTSNode:
    def __init__(self, state, parent: Optional["MCTSNode"] = None, action=None, player_to_move_color=None):
        self.state = state              # the board state the node represents
        self.parent = parent            # the parent node
        self.action = action            # the action used to reach this node

        self.player_to_move_color = player_to_move_color           # the color of the player that is going to move next

        self.children: list[MCTSNode] = []

        self.untried_actions = []

        self.playout_num = 0
        self.reward_num = 0.0         # the number of wins from root player's perspective (our agent's perspective)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

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

def select(node: MCTSNode, total_turn_count_sm: int, seen_states_sm: SeenStates) -> MCTSNode:
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
        total_turn_count_sm += 1
        record_state(seen_states_sm, node.state, node.player_to_move_color)

    return node


# {Expansion}
def expand(node: MCTSNode, player_to_move_color: PlayerColor, total_turn_count_sm: int, seen_states_sm: SeenStates) -> MCTSNode:
    """
    Expand a child node of the given leaf (parent) node by one untried action from this given leaf (parent) node.
    """

    # Check whether the parent has reached a terminal state--defensive--also done in the main logic
    if is_terminal(node.state, total_turn_count_sm, seen_states_sm, node.player_to_move_color):
        return node

    # Step 1: Get one of the possible untried legal actions--future improvement: ordering--based on the evaluation score
    # Get all its legal actions if this is the first time the node being expanded
    if node.untried_actions is None:
        node.untried_actions = get_legal_actions(
            node.board,
            node.player_to_move,
            total_turn_count_sm
        )

    evaluations = {}
    for action in node.untried_actions:
        if is_meaningless_cascade(node.state, player_to_move_color, action):
            continue
        # ss1: Apply the action
        new_board = copy_state(node.state)
        apply_action(new_board, player_to_move_color, action)

        # ss2: Evaluate the new board state after applying this action
        evaluations[action] = evaluate_new(new_board, player_to_move_color, total_turn_count_sm)

    # ss3: Sort all the actions by their evaluation scores
    sorted_evaluations = dict(sorted(evaluations.items(), key=lambda item: item[1], reverse=True))

    action = next(iter(sorted_evaluations))

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
def playout(board: dict[Coord, CellState], my_color: PlayerColor, player_to_move_color: PlayerColor, total_turn_count_sm: int, seen_states_sm: SeenStates, rollout_depth=30) -> float:
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

        # Defensive check
        if not legal_actions:
            break

        # Step 2: Choose the action with the highest evaluation score
        evaluations = {}
        for action in legal_actions:
            if is_meaningless_cascade(current_board, current_player_color, action):
                continue
            # ss1: Apply the action
            new_board = copy_state(current_board)
            apply_action(new_board, current_player_color, action)

            # ss2: Evaluate the new board state after applying this action
            evaluations[action] = evaluate(new_board, current_player_color)

        # ss3: Sort all the actions by their evaluation scores
        sorted_evaluations = dict(sorted(evaluations.items(), key=lambda item: item[1], reverse=True))

        action = next(iter(sorted_evaluations))

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

def get_score_for_playout_result(current_board: dict[Coord, CellState], my_color: PlayerColor, current_player_color: PlayerColor, current_total_turn_count: int, current_seen_states: SeenStates) -> float:
    """
    Return reward from our agent's perspective.
    --This should be updated to be more efficient
    """

    my_score = 0
    enemy_score = 0

    for cell in current_board.values():
        if cell.color == my_color:
            my_score += cell.height
        else:
            enemy_score += cell.height

    if is_terminal(current_board, current_total_turn_count, current_seen_states, current_player_color):
        # Case 1: In the current stopping state, our agent has won--it has eliminated all the opponent's stacks or our agent has more tokens on the board
        if my_score > 0 and enemy_score == 0:
            return 1.0
        elif my_score > enemy_score:
            return 1.0
        # Case 2: A draw
        elif my_score == enemy_score:
            return 0.5
        # Case 3: A lose
        else:
            return 0.0
    else:
        # Case 4: The current board state is favorable, even if we haven't reached an terminal state of the game
        if my_score > enemy_score:
            return 1.0
        # Case 5: The current board state is not favorable
        elif my_score < enemy_score:
            return 0.0
        # Case 6: The current board state is intermediate, hard to tell good or not
        else:
            return 0.5


# {Backpropagation}
def backpropagate(node: MCTSNode, reward: float, player_to_move_color: PlayerColor) -> None:
    """
    Update visits and rewards from the newly expanded and "evaluated" child node back to the root.
    """

    # Traverse from the newly simulated child node back to the root of the tree
    # Update the nodes in this path
    while node is not None:
        node.playout_num += 1
        # For nodes representing the same agent to play next
        if node.player_to_move_color == player_to_move_color:
            node.reward_num += reward
        # For nodes representing the opponent agent to play next
        else:
            node.reward_num += 1.0 - reward

        # Keep traversing
        node = node.parent