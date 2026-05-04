# This file contains the logic about the game play-strategy for the play phase.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N, Action
from .rules import get_legal_actions
from .evaluation_play import evaluate
from .rules import apply_action
from .types import SeenStates
from .helper import encode_state, record_state


# ----------------------------
# Implementation of MCTS
# ----------------------------

