from enum import Enum

from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N
from .helper import get_Manhattan_distance, get_same_direction, successful_cascade, is_adjacent, get_distance_to_edge_shortest, get_all_distance_to_opponent, get_total_height


class BoardState(Enum):
    COMPACT_ALIGNMENT = 1
    OPPONENT_SCATTERED = 2
    EDGE_CORNER_PRESSURE = 3
    PLAYER_SCARCITY = 4
    # OPPONENT_FEW_REMAIN = 5
    ATTACKABLE_OPPONENT_FAR = 6
    HAS_IMMEDIATE_AFFECT = 7
    BALANCED = 8
    IN_OUR_FAVOUR = 9

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

    return get_Manhattan_distance(coord_outer, coord_inner) >= 6

def no_player_between (coord_a: Coord, coord_b: Coord, player_stacks: list[tuple[Coord, CellState]]
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

# A
def is_opponent_aligned (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> bool:
    dense_pair_count = 0
    
    for i, (coord_a, _) in enumerate(opponent_stacks):
        for j in range(i + 1, len(opponent_stacks)):
            coord_b, _ = opponent_stacks[j]

            if is_dense(coord_a, coord_b) and no_player_between(coord_a, coord_b, player_stacks):
                dense_pair_count += 1
                if dense_pair_count >= 2:
                    break

        if dense_pair_count >= 2:
            return True

    return False

# B
def is_opponent_scattered (opponent_stacks: list[tuple[Coord, CellState]]) -> bool:
    scatter_pair_count = 0
    for i, (coord_a, _) in enumerate(opponent_stacks):
        for j in range(i + 1, len(opponent_stacks)):
            coord_b, _ = opponent_stacks[j]
            if is_scatter(coord_a, coord_b):
                scatter_pair_count += 1
                if scatter_pair_count >= 3:
                    break
        if scatter_pair_count >= 3:
            break

    if scatter_pair_count >= 3:
        return True
    
    return False

# C
def is_opponent_corner_edge_pressure (opponent_stacks: list[tuple[Coord, CellState]]) -> bool:
    pressure_num = 0
    for coord_opponent, _ in opponent_stacks:
        if is_pressure(coord_opponent):
            pressure_num += 1
            if pressure_num >= 2:
                return True

    return False

# F
def is_attackable_far (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> bool:
    meaningful_dists = get_all_distance_to_opponent(opponent_stacks, player_stacks)

    far_count = 0
    for dist in meaningful_dists:
        if dist >= 4:
            far_count += 1
    
    if far_count >= 2:
        return True
    
    return False

# G
def has_immediate_elimination (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> bool:
    for coord_player, state_player in player_stacks:
        for coord_opponent, state_opponent in opponent_stacks:
            if is_adjacent(coord_player, coord_opponent) and state_player.height >= state_opponent.height:
                return True
            possible_direction = get_same_direction(coord_player, coord_opponent)
            if state_player.height >= 2 and possible_direction is not None:
                if successful_cascade(board, coord_player, state_player, coord_opponent, possible_direction):
                    return True
    return False

# H
def is_balanced(opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> bool:
    opponent_total_height = get_total_height(opponent_stacks)
    player_total_height = get_total_height(player_stacks)
    
    # If either side has only one stack, do not treat it as balanced
    if len(opponent_stacks) == 1 or len(player_stacks) == 1:
        return False
    
    # First, Must have same total height
    if opponent_total_height != player_total_height:
        return False

    # Second, The difference between the stack number of the two sides should be less than one
    stack_num_diff = abs(len(opponent_stacks) - len(player_stacks))
    if stack_num_diff > 1:
        return False

    return True

# I
def is_in_our_favour (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> bool:
    # First, the opponent has no more than stacks on the board
    if len(opponent_stacks) > len(player_stacks):
        return False

    # Second, our total height is greater the opponent total height of stacks
    opponent_total_height = get_total_height(opponent_stacks)
    player_total_height = get_total_height(player_stacks)
    if opponent_total_height >= player_total_height:
        return False

    return True


# Detect the current state of the board, this is used for weight adjustment
def detect_board_state (
    board: dict[Coord, CellState],
    opponent_stacks: list[tuple[Coord, CellState]],
    player_stacks: list[tuple[Coord, CellState]]
) -> list[BoardState]:
    detected_state: list[BoardState] = []

    # A: Opponent stacks are close and lined up
    is_dense = False
    if is_opponent_aligned(opponent_stacks, player_stacks):
        detected_state.append(BoardState.COMPACT_ALIGNMENT)
        is_dense = True

    # B: Opponent stacks are spread out--this is opposite of A
    if is_opponent_scattered(opponent_stacks) and not is_dense:
        detected_state.append(BoardState.OPPONENT_SCATTERED)

    # C: Opponent is building pressure on edges/corners
    if is_opponent_corner_edge_pressure (opponent_stacks):
        detected_state.append(BoardState.EDGE_CORNER_PRESSURE)

    # D: We are behind in stack count
    if len(player_stacks) - len(opponent_stacks) <= -3:
        detected_state.append(BoardState.PLAYER_SCARCITY)

    # E: The opponent is behind in stack count--the board is favrible for us already
    # Not Useful might be--this case should be included in "is in our favour"
    #if len(player_stacks) - len(opponent_stacks) >= 3 or len(opponent_stacks) == 1:
        #detected_state.append(BoardState.OPPONENT_FEW_REMAIN)

    # E: The attackable opponents are far from us
    if is_attackable_far(opponent_stacks, player_stacks):
        detected_state.append(BoardState.ATTACKABLE_OPPONENT_FAR)

    # F: The board state has immediate elimination with one of the actions it's going to be applied
    if has_immediate_elimination(board, opponent_stacks, player_stacks):
        detected_state.append(BoardState.HAS_IMMEDIATE_AFFECT)

    # G: The board state is balanced: similar total stack height and similar stack number
    if is_balanced(opponent_stacks, player_stacks):
        detected_state.append(BoardState.BALANCED)

    # H: The board state is in our player's favour
    if is_in_our_favour(opponent_stacks, player_stacks):
        detected_state.append(BoardState.IN_OUR_FAVOUR)

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

# Find the total distance between each stack and the nearest edge
def get_total_dist_to_edge (stacks: list[tuple[Coord, CellState]]) -> float:
    total_nearest_edge_distance = 0
    for coord, _ in stacks:
        dist_edge_nearest = get_distance_to_edge_shortest(coord)
        total_nearest_edge_distance += dist_edge_nearest
    
    return total_nearest_edge_distance
        
