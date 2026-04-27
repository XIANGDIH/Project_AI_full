# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .rules import apply_action, get_legal_actions
from .evaluation import choose_coord_placement_phase


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
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

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
                    print("Testing: RED is playing a PLACE action")
                    
                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    return PlaceAction(best_placing_coord)
                case PlayerColor.BLUE:
                    print("Testing: BLUE is playing a PLACE action")

                    best_placing_coord = choose_coord_placement_phase(self._board, legal_actions, self._color, self._turn_count)
                    return PlaceAction(best_placing_coord)

        # During play phase
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")
                return MoveAction(Coord(0, 0), Direction.Down)
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a MOVE action")
                return MoveAction(Coord(7, 0), Direction.Up)
            
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
        
