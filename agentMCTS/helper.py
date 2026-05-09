# This file contains helpers for evaluations


from itertools import product
import random

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from .types import SeenStates


# {Basic}
def get_Manhattan_distance (coord_a: Coord, coord_b: Coord) -> float:
    return abs(coord_b.r - coord_a.r) + abs(coord_b.c - coord_a.c)

def is_in_board (coord: Coord) -> bool:
    return 0 <= coord < BOARD_N and 0 <= coord < BOARD_N

def is_in_same_line (coord_a: Coord, coord_b: Coord) -> bool:
    same_line = (coord_a.r == coord_b.r) or (coord_a.c == coord_b.c)
    return same_line

def is_adjacent (coord_a: Coord, coord_b: Coord) -> bool:
    same_line = (coord_a.r == coord_b.r) or (coord_a.c == coord_b.c)
    # On the same line
    if not same_line:
        return False
    
    # Distance is 1
    return get_Manhattan_distance(coord_a, coord_b) == 1

def is_on_edge (coord: Coord) -> bool:
    return (
        coord.r == 0 or coord.r == BOARD_N - 1 or
        coord.c == 0 or coord.c == BOARD_N - 1
    )

def get_distance_to_edge_shortest (coord: Coord) -> float:
    dist_top_edge = coord.r - 0
    dist_bottom_edge = 7 - coord.r
    dist_left_edge = coord.c - 0
    dist_right_edge = 7 - coord.c

    return min(dist_top_edge, dist_bottom_edge, dist_left_edge, dist_right_edge)

def get_distance_to_centre (coord: Coord) -> float:
    centre_r = 3.5
    centre_c = 3.5
    return abs(coord.r - centre_r) + abs(coord.c - centre_c)

def get_closest_to_centre (coords: list[Coord]) -> Coord | None:
    closest_coordinate = None
    closest_to_centre_distance = 10.0
    for coord in coords:
        dist = get_distance_to_centre(coord)
        if dist < closest_to_centre_distance:
            closest_to_centre_distance = dist
            closest_coordinate = coord
    
    return closest_coordinate

def get_edge_labels (coord: Coord) -> list[str]:
    labels = []

    if coord.r == 0:
        labels.append("top")
    if coord.r == BOARD_N - 1:
        labels.append("bottom")
    if coord.c == 0:
        labels.append("left")
    if coord.c == BOARD_N - 1:
        labels.append("right")

    return labels

def get_corner_label (coord: Coord) -> str | None:
    if coord.r == 0 and coord.c == 0:
        return "top_left"

    if coord.r == 0 and coord.c == BOARD_N - 1:
        return "top_right"

    if coord.r == BOARD_N - 1 and coord.c == 0:
        return "bottom_left"

    if coord.r == BOARD_N - 1 and coord.c == BOARD_N - 1:
        return "bottom_right"

    return None

# Whether the specific victim and attacker stack pair is in the same line, if it is get the direction of attack for the attacker
def get_same_direction (coord_attacker: Coord, coord_victim: Coord) -> Direction | None:
    if coord_attacker == coord_victim:
        return None

    if coord_attacker.r == coord_victim.r:
        return Direction.Right if coord_victim.c > coord_attacker.c else Direction.Left

    if coord_attacker.c == coord_victim.c:
        return Direction.Down if coord_victim.r > coord_attacker.r else Direction.Up

    return None

def get_opposite_direction (direction: Direction) -> Direction:
    match direction:
        case Direction.Up:
            return Direction.Down
        case Direction.Down:
            return Direction.Up
        case Direction.Left:
            return Direction.Right
        case Direction.Right:
            return Direction.Left
        case _:
            print("ERROR: Finding the opposite direction")
            return None

# {Related to score calculations}
# Count how many stacks (no matter Blue or Red) are in between of the given pair
def count_stacks_between (coord_a: Coord, coord_b: Coord, occupied: set[Coord]) -> int:
    # Same row
    if coord_a.r == coord_b.r:
        row = coord_a.r
        c_min = min(coord_a.c, coord_b.c)
        c_max = max(coord_a.c, coord_b.c)

        count = 0
        for c in range(c_min + 1, c_max):
            if Coord(row, c) in occupied:
                count += 1
        return count

    # Same column
    if coord_a.c == coord_b.c:
        col = coord_a.c
        r_min = min(coord_a.r, coord_b.r)
        r_max = max(coord_a.r, coord_b.r)

        count = 0
        for r in range(r_min + 1, r_max):
            if Coord(r, col) in occupied:
                count += 1
        return count

    return 0

# Whether the new coordinate is off the board
def is_off_board_after(coord_old: Coord, step: int, direction: Direction, stack_in_between_num: int) -> bool:
    dr, dc = direction.value

    push_distance = step - stack_in_between_num

    # Defensive check
    if push_distance <= 0:
        return False

    coord_new_r = coord_old.r + dr * push_distance
    coord_new_c = coord_old.c + dc * push_distance

    return not (0 <= coord_new_r < BOARD_N and 0 <= coord_new_c < BOARD_N)

def is_in_direction_path(
    attacker: Coord,
    victim: Coord,
    direction: Direction
) -> bool:
    if direction == Direction.Up:
        return victim.c == attacker.c and victim.r < attacker.r

    if direction == Direction.Down:
        return victim.c == attacker.c and victim.r > attacker.r

    if direction == Direction.Left:
        return victim.r == attacker.r and victim.c < attacker.c

    if direction == Direction.Right:
        return victim.r == attacker.r and victim.c > attacker.c

    return False


# {Performance}
# Whether the cascade action of the specific stack is successful (it can eliminate the corresponding opponent stack we are looking at)
# in terms of the given pair
def successful_cascade (board: dict[Coord, CellState], coord_attacker: Coord, state_attacker: CellState, coord_victim: Coord, direction: Direction) -> bool:
    is_successful = False

    # Whether the height of the attacking stack we are looking at is >= 2
    if state_attacker.height < 2:
        return False
    
    if not is_in_direction_path(coord_attacker, coord_victim, direction):
        return False
    
    step = state_attacker.height
    board_stack_on = set(board.keys())
    stack_inbetween_num = count_stacks_between(coord_attacker, coord_victim, board_stack_on)

    # Get the new position of Blue stack we are looking at 
    if is_off_board_after(coord_victim, step, direction, stack_inbetween_num):
        is_successful = True

    return is_successful

# Whether the cascade action of the specific stack is meaningful (there is at least one opponent stack at the same direction of the cascade)
# This is a weaker version
def meaningful_cascade (coord_attacker: Coord, state_attacker: CellState, stacks_victim: list[tuple[Coord, CellState]], direction: Direction) -> bool:

    # Whether the height of the attacking stack we are looking at is >= 2
    if state_attacker.height < 2:
        return False
    
    for coord_victim, _ in stacks_victim:
        if is_in_direction_path(coord_attacker, coord_victim, direction):
            #print("DEBUG: Here\n")
            return True
    
    return False

def encode_state(board: dict[Coord, CellState]) -> tuple:
    return tuple(sorted(
        (coord.r, coord.c, cell.color, cell.height)
        for coord, cell in board.items()
    ))


def record_state(
    seen_states: SeenStates,
    board: dict[Coord, CellState],
    color: PlayerColor
) -> None:
    encoded = encode_state(board)

    if encoded in seen_states:
        seen_count, previous_color = seen_states[encoded]
        seen_states[encoded] = (seen_count + 1, previous_color)
    else:
        seen_states[encoded] = (1, color)
