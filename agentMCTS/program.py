# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .rules import apply_action, get_legal_actions
from .evaluation_placement import choose_coord_placement_phase
from .types import SeenStates
from .helper import encode_state, record_state
from .search import mcts_choose_action
from .evaluation_play import get_weights_for_color, get_feature_breakdown
from .logging_utils import log_event


verbose: bool = False
DEPTH_SEARCH = 3

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        # How many turns I have played so far.
        self._turn_count = 0
        self._board: dict[Coord, CellState] = {}
        # How many turns have happened in total (me + opponent).
        self._total_turn_count = 0
        match color:
            case PlayerColor.RED:
                if verbose:
                    print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                if verbose:
                    print("Testing: I am playing as BLUE")

        # A dictionary to keep track of all seen states
        self._seen_states: SeenStates = {}
        record_state(self._seen_states, self._board, self._color)
        self._weights = get_weights_for_color(self._color)

        log_event(
            "agent_init",
            {
                "player": str(self._color),
                "depth_search": DEPTH_SEARCH,
                "weights": self._weights,
            },
        )

    def _action_to_text(self, action: Action) -> str:
        match action:
            case PlaceAction(coord):
                return f"PLACE({coord.r}-{coord.c})"
            case MoveAction(coord, direction):
                return f"MOVE({coord.r}-{coord.c},{direction})"
            case EatAction(coord, direction):
                return f"EAT({coord.r}-{coord.c},{direction})"
            case CascadeAction(coord, direction):
                return f"CASCADE({coord.r}-{coord.c},{direction})"
            case _:
                return str(action)

    def _count_stacks(self) -> tuple[int, int]:
        red_count = 0
        blue_count = 0

        for _, cell_state in self._board.items():
            if cell_state.color == PlayerColor.RED:
                red_count += 1
            else:
                blue_count += 1

        return red_count, blue_count

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        # Below we have hardcoded actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        # During placement phase (first 8 turns total, 4 per player)
        if self._turn_count < 4:
            # Step 1: Get all legal actions
            legal_actions = get_legal_actions(self._board, self._color, self._turn_count)
            #print("DEBUG: Here")
            #print(legal_actions)
            #print("\n")
            # Step 2: Choose the best coordinate to place our stack
            match self._color:
                case PlayerColor.RED:
                    if verbose:
                        print("Testing: RED is playing a PLACE action")
                    
                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    action = PlaceAction(best_placing_coord)
                    log_event(
                        "turn_decision",
                        {
                            "player": str(self._color),
                            "phase": "placement",
                            "my_turn_index": self._turn_count + 1,
                            "total_turn_index": self._total_turn_count + 1,
                            "legal_action_count": len(legal_actions),
                            "chosen_action": self._action_to_text(action),
                        },
                    )
                    return action
                case PlayerColor.BLUE:
                    if verbose:
                        print("Testing: BLUE is playing a PLACE action")

                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    action = PlaceAction(best_placing_coord)
                    log_event(
                        "turn_decision",
                        {
                            "player": str(self._color),
                            "phase": "placement",
                            "my_turn_index": self._turn_count + 1,
                            "total_turn_index": self._total_turn_count + 1,
                            "legal_action_count": len(legal_actions),
                            "chosen_action": self._action_to_text(action),
                        },
                    )
                    return action

        # During play phase
        feature_map = get_feature_breakdown(
            self._board,
            self._color,
            self._total_turn_count,
            self._weights,
        )
        log_event(
            "feature_snapshot",
            {
                "player": str(self._color),
                "phase": "play",
                "my_turn_index": self._turn_count + 1,
                "total_turn_index": self._total_turn_count + 1,
                "features": feature_map,
                "weights": self._weights,
            },
        )

        match self._color:
            case PlayerColor.RED:
                if verbose:
                    print("Testing: RED is playing a MOVE action")
                action = mcts_choose_action(
                    self._board,
                    self._color,
                    self._total_turn_count,
                    self._seen_states,
                    self._weights,
                )
                log_event(
                    "turn_decision",
                    {
                        "player": str(self._color),
                        "phase": "play",
                        "my_turn_index": self._turn_count + 1,
                        "total_turn_index": self._total_turn_count + 1,
                        "chosen_action": self._action_to_text(action),
                        "weights": self._weights,
                    },
                )
                return action
            case PlayerColor.BLUE:
                if verbose:
                    print("Testing: BLUE is playing a MOVE action")
                action = mcts_choose_action(
                    self._board,
                    self._color,
                    self._total_turn_count,
                    self._seen_states,
                    self._weights,
                )
                log_event(
                    "turn_decision",
                    {
                        "player": str(self._color),
                        "phase": "play",
                        "my_turn_index": self._turn_count + 1,
                        "total_turn_index": self._total_turn_count + 1,
                        "chosen_action": self._action_to_text(action),
                        "weights": self._weights,
                    },
                )
                return action
            
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._total_turn_count += 1

        if color == self._color:
            self._turn_count += 1

        # There are four possible action types: PLACE, MOVE, EAT, and CASCADE.
        # Below we check which type of action was played and print out the
        # details of the action for demonstration purposes. You should replace
        # this with your own logic to update your agent's internal game state.
        apply_action(self._board, color, action)

        record_state(self._seen_states, self._board, color)

        red_count, blue_count = self._count_stacks()
        log_event(
            "turn_update",
            {
                "player_who_moved": str(color),
                "total_turn_index": self._total_turn_count,
                "action": self._action_to_text(action),
                "red_stack_count": red_count,
                "blue_stack_count": blue_count,
                "seen_state_count": len(self._seen_states),
            },
        )
