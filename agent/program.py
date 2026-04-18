# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction


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
            match self._color:
                case PlayerColor.RED:
                    print("Testing: RED is playing a PLACE action")
                    return PlaceAction(Coord(0, self._turn_count))
                case PlayerColor.BLUE:
                    print("Testing: BLUE is playing a PLACE action")
                    return PlaceAction(Coord(7, self._turn_count))

        # During play phase
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")
                return MoveAction(Coord(0, 0), Direction.Down)
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a MOVE action")
                return MoveAction(Coord(7, 0), Direction.Up)

    def get_legal_actions(self) -> list[Action]:
        """
        Generate all legal actions for the current agent from self._board.
        This is a baseline legal-action generator (no scoring/ordering).
        """
        legal_actions: list[Action] = []

        # ----------------------------
        # Placement phase (first 8 turns in total, 4 per player)
        # ----------------------------
        if self._total_turn_count < 8:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)

                    # Cannot place on an occupied cell.
                    if coord in self._board:
                        continue

                    # After the first placement turn, cannot place adjacent to opponent.
                    if self._total_turn_count > 0 and self._is_adjacent_to_opponent(coord, self._color):
                        continue

                    legal_actions.append(PlaceAction(coord))

            return legal_actions

        # ----------------------------
        # Play phase
        # ----------------------------
        for coord, cell_state in self._board.items():
            # Only generate actions for our own stacks.
            if cell_state.color != self._color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                target_r = coord.r + direction.r
                target_c = coord.c + direction.c

                # Move/Eat need an adjacent in-bounds target.
                if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                    continue

                target_coord = Coord(target_r, target_c)
                target_state = self._board.get(target_coord)

                # MOVE-relocate to empty target.
                if target_state is None:
                    legal_actions.append(MoveAction(coord, direction))
                else:
                    # MOVE-merge onto friendly stack.
                    if target_state.color == self._color:
                        legal_actions.append(MoveAction(coord, direction))
                    # EAT if enemy stack height <= attacker height.
                    elif cell_state.height >= target_state.height:
                        legal_actions.append(EatAction(coord, direction))

            # CASCADE is legal if stack height >= 2 (for any cardinal direction).
            if cell_state.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    legal_actions.append(CascadeAction(coord, direction))

        return legal_actions

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
        match action:
            case PlaceAction(coord):
                self._board[coord] = CellState(color, INITIAL_STACK_HEIGHT)
                print(f"Testing: {color} played PLACE action at {coord}")
            case MoveAction(coord, direction):
                source_coord = coord
                source_state = self._board[source_coord]
                target_coord = coord + direction

                # If target has a friendly stack, merge heights; otherwise relocate.
                if target_coord in self._board:
                    target_state = self._board[target_coord]
                    self._board[target_coord] = CellState(
                        color,
                        source_state.height + target_state.height
                    )
                else:
                    self._board[target_coord] = source_state

                # Original cell becomes empty (removed from sparse board dict).
                self._board.pop(source_coord, None)
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case EatAction(coord, direction):
                source_coord = coord
                source_state = self._board[source_coord]
                target_coord = coord + direction

                # Capture: attacker stack moves onto target and replaces it.
                self._board[target_coord] = source_state

                # Original cell becomes empty (removed from sparse board dict).
                self._board.pop(source_coord, None)
                print(f"Testing: {color} played EAT action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case CascadeAction(coord, direction):
                source_coord = coord
                source_state = self._board[source_coord]
                self._board.pop(source_coord, None)

                # Cascade each token one-by-one along the chosen direction.
                for step in range(1, source_state.height + 1):
                    target_r = source_coord.r + step * direction.r
                    target_c = source_coord.c + step * direction.c

                    # Tokens landing out of bounds are discarded.
                    if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                        continue

                    target_coord = Coord(target_r, target_c)

                    # If occupied, push that stack first.
                    if target_coord in self._board:
                        self._push_stack(target_coord, direction)

                    # Then place one token of the acting player.
                    self._board[target_coord] = CellState(color, 1)
                print(f"Testing: {color} played CASCADE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def _push_stack(self, coord: Coord, direction: Direction) -> None:
        """
        Push one whole stack forward by one cell.
        If another stack is in front, push it first recursively.
        If pushed out of bounds, the stack is removed.
        """
        if coord not in self._board:
            return

        current_stack = self._board[coord]
        next_r = coord.r + direction.r
        next_c = coord.c + direction.c

        # Remove current position first; this cell will be vacated after push.
        self._board.pop(coord, None)

        # Pushed off board -> eliminated.
        if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
            return

        next_coord = Coord(next_r, next_c)

        if next_coord in self._board:
            self._push_stack(next_coord, direction)

        self._board[next_coord] = current_stack

    def _is_adjacent_to_opponent(self, coord: Coord, color: PlayerColor) -> bool:
        """
        Return True if this coord is adjacent (cardinal) to any opponent stack.
        Used by placement-phase legality check.
        """
        opponent = color.opponent

        for direction in CARDINAL_DIRECTIONS:
            neighbor_r = coord.r + direction.r
            neighbor_c = coord.c + direction.c

            if not (0 <= neighbor_r < BOARD_N and 0 <= neighbor_c < BOARD_N):
                continue

            neighbor_coord = Coord(neighbor_r, neighbor_c)
            neighbor_state = self._board.get(neighbor_coord)

            if neighbor_state is not None and neighbor_state.color == opponent:
                return True

        return False
