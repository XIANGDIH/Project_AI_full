# This file contians realizations about the rules of the Cascade game

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

def get_legal_actions(
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int) -> list[Action]:
    """
    Generate all legal actions for the current agent from self._board.
    This is a baseline legal-action generator (no scoring/ordering).
    """
    legal_actions: list[Action] = []

    # ----------------------------
    # Placement phase (first 8 turns in total, 4 per player)
    # ----------------------------
    if total_turn_count < 4:
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)

                # Cannot place on an occupied cell.
                if coord in board:
                    continue

                # After the first placement turn, cannot place adjacent to opponent.
                if board is not None and is_adjacent_to_opponent(board, coord, color):
                    continue

                legal_actions.append(PlaceAction(coord))

        return legal_actions

    # ----------------------------
    # Play phase
    # ----------------------------
    for coord, cell_state in board.items():
        # Only generate actions for our own stacks.
        if cell_state.color != color:
            continue

        for direction in CARDINAL_DIRECTIONS:
            target_r = coord.r + direction.r
            target_c = coord.c + direction.c

            # Move/Eat need an adjacent in-bounds target.
            if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                continue

            target_coord = Coord(target_r, target_c)
            target_state = board.get(target_coord)

            # MOVE-relocate to empty target.
            if target_state is None:
                legal_actions.append(MoveAction(coord, direction))
            else:
                # MOVE-merge onto friendly stack.
                if target_state.color == color:
                    legal_actions.append(MoveAction(coord, direction))
                # EAT if enemy stack height <= attacker height.
                elif cell_state.height >= target_state.height:
                    legal_actions.append(EatAction(coord, direction))

        # CASCADE is legal if stack height >= 2 (for any cardinal direction).
        if cell_state.height >= 2:
            for direction in CARDINAL_DIRECTIONS:
                legal_actions.append(CascadeAction(coord, direction))

    return legal_actions

def push_stack(board: dict[Coord, CellState], coord: Coord, direction: Direction) -> None:
    """
    Push one whole stack forward by one cell.
    If another stack is in front, push it first recursively.
    If pushed out of bounds, the stack is removed.
    """
    if coord not in board:
        return

    current_stack = board[coord]
    next_r = coord.r + direction.r
    next_c = coord.c + direction.c

    # Remove current position first; this cell will be vacated after push.
    board.pop(coord, None)

    # Pushed off board -> eliminated.
    if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
        return

    next_coord = Coord(next_r, next_c)

    # Recursively push the stacks in the way.
    if next_coord in board:
        push_stack(board, next_coord, direction)

    board[next_coord] = current_stack

def is_adjacent_to_opponent(board: dict[Coord, CellState], coord: Coord, color: PlayerColor) -> bool:
    """
    Return True if this coord is adjacent (cardinal) to any opponent stack.
    Used by placement-phase legality check.
    """
    opponent = color.opponent

    # Check all of the four directions.
    for direction in CARDINAL_DIRECTIONS:
        neighbor_r = coord.r + direction.r
        neighbor_c = coord.c + direction.c

        # If the neighbor at the specific direction is outside of the board boundary, skip it.
        if not (0 <= neighbor_r < BOARD_N and 0 <= neighbor_c < BOARD_N):
            continue

        neighbor_coord = Coord(neighbor_r, neighbor_c)
        neighbor_state = board.get(neighbor_coord)

        if neighbor_state is not None and neighbor_state.color == opponent:
            return True

    return False

def apply_action(
    board: dict[Coord, CellState],
    color: PlayerColor,
    action: Action,
    verbose: bool = True,
) -> None:
    match action:
        case PlaceAction(coord):
            board[coord] = CellState(color, INITIAL_STACK_HEIGHT)

            if verbose:
                print(f"Testing: {color} played PLACE action at {coord}")
        
        case MoveAction(coord, direction):
            source_coord = coord
            source_state = board[source_coord]
            target_coord = coord + direction

            # If target has a friendly stack, merge heights; otherwise relocate.
            if target_coord in board:
                target_state = board[target_coord]

                # Defensive Check
                if target_state.color != color:
                    raise ValueError("MOVE cannot move onto opponent stack")

                board[target_coord] = CellState(
                    color,
                    source_state.height + target_state.height
                )
            else:
                board[target_coord] = source_state

            # Original cell becomes empty (removed from sparse board dict).
            board.pop(source_coord, None)
            
            if verbose:
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")

        case EatAction(coord, direction):
            source_coord = coord
            source_state = board[source_coord]
            target_coord = coord + direction

            # Capture: attacker stack moves onto target and replaces it.
            board[target_coord] = source_state

            # Original cell becomes empty (removed from sparse board dict).
            board.pop(source_coord, None)

            if verbose:
                print(f"Testing: {color} played EAT action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")


        case CascadeAction(coord, direction):
            source_coord = coord
            source_state = board[source_coord]
            board.pop(source_coord, None)

            # Cascade each token one-by-one along the chosen direction.
            for step in range(1, source_state.height + 1):
                target_r = source_coord.r + step * direction.r
                target_c = source_coord.c + step * direction.c

                # Tokens landing out of bounds are discarded.
                if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                    continue

                target_coord = Coord(target_r, target_c)

                # If occupied, push that stack first.
                if target_coord in board:
                    push_stack(board, target_coord, direction)

                # Then place one token of the acting player.
                board[target_coord] = CellState(color, 1)

            if verbose:
                print(f"Testing: {color} played CASCADE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")

        case _:
            raise ValueError(f"Unknown action type: {action}")
