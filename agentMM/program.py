# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent


from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .rules import apply_action, get_legal_actions
from .evaluation_placement import choose_coord_placement_phase
from .search import choose_action
from .evaluation_play import get_feature_breakdown, get_weights_for_color
from .types import SeenStates
from .helper import encode_state, record_state
from .logging_utils import log_json


DEPTH_SEARCH = 3

class Agent:
    """
    This is the main entry point for the agent.
    The referee calls methods in this class during the game.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This runs once when the referee creates the agent.
        Put all setup work here.
        """
        self._color = color
        # Number of turns this agent has played.
        self._turn_count = 0
        self._board: dict[Coord, CellState] = {}
        # Number of total turns so far (me + opponent).
        self._total_turn_count = 0
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

        # Track repeated board states for repetition checks.
        self._seen_states: SeenStates = {}
        record_state(self._seen_states, self._board, self._color)
        self._weights = get_weights_for_color(self._color)

        init_data = {
            "player": str(self._color),
            "depth_search": DEPTH_SEARCH,
            "weights": self._weights,
        }
        log_json("agent_init", init_data)

    def _action_to_text(self, action: Action) -> str:
        """
        Turn an action object into a short string for logs.
        """
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
        """
        Count how many stacks RED and BLUE currently have.
        """
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
        The referee calls this on our turn.
        We must return exactly one valid action.
        """

        # Placement phase: first 4 turns for each player.
        if self._turn_count < 4:
            # First, collect all legal actions.
            legal_actions = get_legal_actions(self._board, self._color, self._turn_count)
            #print("DEBUG: Here")
            #print(legal_actions)
            #print("\n")
            # Then pick where to place.
            match self._color:
                case PlayerColor.RED:
                    print("Testing: RED is playing a PLACE action")
                    
                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    action = PlaceAction(best_placing_coord)
                    decision_data = {
                        "player": str(self._color),
                        "phase": "placement",
                        "my_turn_index": self._turn_count + 1,
                        "total_turn_index": self._total_turn_count + 1,
                        "legal_action_count": len(legal_actions),
                        "chosen_action": self._action_to_text(action),
                    }
                    log_json("turn_decision", decision_data)
                    return action
                case PlayerColor.BLUE:
                    print("Testing: BLUE is playing a PLACE action")

                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    action = PlaceAction(best_placing_coord)
                    decision_data = {
                        "player": str(self._color),
                        "phase": "placement",
                        "my_turn_index": self._turn_count + 1,
                        "total_turn_index": self._total_turn_count + 1,
                        "legal_action_count": len(legal_actions),
                        "chosen_action": self._action_to_text(action),
                    }
                    log_json("turn_decision", decision_data)
                    return action

        # Play phase starts here.
        feature_map = get_feature_breakdown(
            self._board,
            self._color,
            self._total_turn_count,
            self._weights,
        )
        log_json(
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
                print("Testing: RED is playing a MOVE action")
                action = choose_action(
                    self._board,
                    self._color,
                    DEPTH_SEARCH,
                    self._total_turn_count,
                    self._seen_states,
                    self._weights,
                )
                log_json(
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
                print("Testing: BLUE is playing a MOVE action")
                action = choose_action(
                    self._board,
                    self._color,
                    DEPTH_SEARCH,
                    self._total_turn_count,
                    self._seen_states,
                    self._weights,
                )
                log_json(
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
        The referee calls this after every move.
        We update our local board and state trackers here.
        """
        self._total_turn_count += 1

        if color == self._color:
            self._turn_count += 1

        # Apply the action to our local board copy.
        apply_action(self._board, color, action)

        record_state(self._seen_states, self._board, color)

        red_stack_count, blue_stack_count = self._count_stacks()
        update_data = {
            "player_who_moved": str(color),
            "total_turn_index": self._total_turn_count,
            "action": self._action_to_text(action),
            "red_stack_count": red_stack_count,
            "blue_stack_count": blue_stack_count,
            "seen_state_count": len(self._seen_states),
        }
        log_json("turn_update", update_data)
