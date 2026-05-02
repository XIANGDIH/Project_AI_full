from enum import Enum

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N
from .helper import get_Manhattan_distance, get_same_direction, successful_cascade, is_adjacent


class BoardState(Enum):
    COMPACT_ALIGNMENT = 1
    EDGE_CORNER_PRESSURE = 2
    PLAYER_SCARCITY = 3
    OPPONENT_SCATTERED = 4

# ----------------------------
# Helpers to read board patterns
# ----------------------------

def is_dense (coord_outer: Coord, coord_inner: Coord) -> bool:
    if coord_outer == coord_inner:
        return False

    same_line = (coord_outer.r == coord_inner.r) or (coord_outer.c == coord_inner.c)
    if not same_line:
        return False

    return get_Manhattan_distance(coord_outer, coord_inner) <= 2


def is_scatter (coord_outer: Coord, coord_inner: Coord) -> bool:
    if coord_outer == coord_inner:
        return False

    return get_Manhattan_distance(coord_outer, coord_inner) >= 4


def no_player_between (
    coord_a: Coord,
    coord_b: Coord,
    player_stacks: list[tuple[Coord, CellState]]
) -> bool:
    # Same row
    if coord_a.r == coord_b.r:
        row = coord_a.r
        c_min = min(coord_a.c, coord_b.c)
        c_max = max(coord_a.c, coord_b.c)

        for coord_player, _ in player_stacks:
            if coord_player.r == row and c_min < coord_player.c < c_max:
                return False
        return True

    # Same column
    if coord_a.c == coord_b.c:
        col = coord_a.c
        r_min = min(coord_a.r, coord_b.r)
        r_max = max(coord_a.r, coord_b.r)

        for coord_player, _ in player_stacks:
            if coord_player.c == col and r_min < coord_player.r < r_max:
                return False
        return True

    return False


def is_pressure (coord: Coord) -> bool:
    return (
        coord.r == 0 or coord.r == BOARD_N - 1 or
        coord.c == 0 or coord.c == BOARD_N - 1
    )

# Detect the current state of the board, this is used for weight adjustment
def detect_board_state (
    opponent_stacks: list[tuple[Coord, CellState]],
    player_stacks: list[tuple[Coord, CellState]]
) -> list[BoardState]:
    detected_state: list[BoardState] = []

    # A: Opponent stacks are close and lined up
    dense_pair_count = 0
    for i, (coord_a, _) in enumerate(opponent_stacks):
        for j in range(i + 1, len(opponent_stacks)):
            coord_b, _ = opponent_stacks[j]

            if is_dense(coord_a, coord_b) and no_player_between(coord_a, coord_b, player_stacks):
                dense_pair_count += 1
                if dense_pair_count >= 2:
                    break

        if dense_pair_count >= 2:
            detected_state.append(BoardState.COMPACT_ALIGNMENT)
            break

    # B: Opponent is building pressure on edges/corners
    pressure_num = 0
    for coord_opponent, _ in opponent_stacks:
        if is_pressure(coord_opponent):
            pressure_num += 1
            if pressure_num >= 2:
                detected_state.append(BoardState.EDGE_CORNER_PRESSURE)
                break

    # C: We are behind in stack count
    if len(player_stacks) - len(opponent_stacks) <= -2:
        detected_state.append(BoardState.PLAYER_SCARCITY)

    # D: Opponent stacks are spread out
    scatter_pair_count = 0
    for i, (coord_a, _) in enumerate(opponent_stacks):
        for j in range(i + 1, len(opponent_stacks)):
            coord_b, _ = opponent_stacks[j]
            if is_scatter(coord_a, coord_b):
                scatter_pair_count += 1
                if scatter_pair_count >= 2:
                    break
        if scatter_pair_count >= 2:
            break

    if scatter_pair_count >= 2 and dense_pair_count == 0:
        detected_state.append(BoardState.OPPONENT_SCATTERED)

    return detected_state


# ----------------------------
# Helpers for scoring
# ----------------------------

def is_adjacent_to_opponent (coord_player: Coord, coord_opponent: Coord) -> bool:
    dr = abs(coord_opponent.r - coord_player.r)
    dc = abs(coord_opponent.c - coord_player.c)
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

def get_threat (
    coord_player: Coord,
    state_player: CellState,
    coord_opponent: Coord,
    state_opponent: CellState,
    board: dict[Coord, CellState],
    state: list[BoardState]
) -> float:
    state_impact_cascade = 0.0
    state_impact_same_direction = 0.0

    if BoardState.COMPACT_ALIGNMENT in state:
        state_impact_cascade = 0.02

    if (
        BoardState.EDGE_CORNER_PRESSURE in state or
        BoardState.PLAYER_SCARCITY in state or
        BoardState.OPPONENT_SCATTERED in state
    ):
        state_impact_same_direction = 0.2

    # Can eat right now
    if is_adjacent(coord_player, coord_opponent) and state_player.height >= state_opponent.height:
        return 0.1

    # Check cascade pressure on this opponent stack
    possible_direction = get_same_direction(coord_player, coord_opponent)
    if state_player.height >= 2 and possible_direction is not None:
        if successful_cascade(board, coord_player, state_player, coord_opponent, possible_direction):
            return 0.1 - state_impact_cascade
        else:
            return 0.3 - state_impact_same_direction

    return 1.0