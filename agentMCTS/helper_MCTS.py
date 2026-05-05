import math
import random
from typing import Optional

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N
from .rules import get_legal_actions, apply_action
from .helper import encode_state
from .types import SeenStates

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

# Node
class MCTSNode:
    def __init__(self, state, parent: Optional["MCTSNode"] = None, action=None):
        self.state = state              # game state at this node
        self.parent = parent            # parent node
        self.action = action            # action used to reach this node

        self.children: list[MCTSNode] = []

        self.untried_actions = get_legal_actions(state)

        self.visits = 0
        self.total_reward = 0.0         # reward from root player's perspective

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        # Need to be rewrited
        return is_terminal(self.state)

    def average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
# {Selection}
# Calculate the UCB score for selection
def uct_score(parent: MCTSNode, child: MCTSNode, exploration_constant=math.sqrt(2)) -> float:
    if child.visits == 0:
        return float("inf")

    exploitation = child.total_reward / child.visits
    exploration = exploration_constant * math.sqrt(
        math.log(parent.visits) / child.visits
    )

    return exploitation + exploration

def select(node: MCTSNode) -> MCTSNode:
    """
    Move down the tree while the node is fully expanded and non-terminal.
    """
    while not node.is_terminal() and node.is_fully_expanded():
        node = max(
            node.children,
            key=lambda child: uct_score(node, child)
        )

    return node

# {Expansion}
def expand(node: MCTSNode) -> MCTSNode:
    """
    Expand one untried action from this node.
    """
    if node.is_terminal():
        return node

    action = node.untried_actions.pop()

    next_state = apply_action(node.state, action)

    child = MCTSNode(
        state=next_state,
        parent=node,
        action=action
    )

    node.children.append(child)

    return child

