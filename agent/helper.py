# This file contains helpers for evaluations
from itertools import product
import random

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

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

# Whether the specific victim and attacker stack pair is in the same direction, if it is get the direction
def get_same_direction (coord_attacker: Coord, coord_victim: Coord) -> Direction | None:
    if coord_attacker == coord_victim:
        return None

    if coord_attacker.r == coord_victim.r:
        return Direction.Right if coord_victim.c > coord_attacker.c else Direction.Left

    if coord_attacker.c == coord_victim.c:
        return Direction.Down if coord_victim.r > coord_attacker.r else Direction.Up

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

# in the casecade path of the opponent stack == if we put a stack at a specific coordinate (on the path), it can be cascaded off
def coord_is_in_cascade_path (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], landing_coord: Coord) -> bool:
    attacker_stacks = opponent_stacks
    coord_victim = landing_coord

    is_in_path = False

    # Check the specific coordinate with all already-placed opponent stacks in the four directions
    for coord_attacker, state_attacker in attacker_stacks:
        for direction in Direction:
            # If placed on the coordinate, a stack can be pushed off by the opponent stack
            if successful_cascade(board, coord_attacker, state_attacker, coord_victim, direction):
                is_in_path = True
                break

    return is_in_path

def in_safe_area (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], landing_coord: Coord) -> bool:
    # Not in the easy-to-be-pushed-off area
    safe_1 = 3<=landing_coord.c<=4 and 1<=landing_coord.r<=6
    safe_2 = 3<=landing_coord.r<=4 and 1<=landing_coord.c<=6

    # Not in the pushed-off areas of the current placed opponent satcks
    safe_3 = not (coord_is_in_cascade_path(board, opponent_stacks, landing_coord))

    return (safe_1 or safe_2) and safe_3

def in_possible_area (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], landing_coord: Coord) -> bool:
    return not (coord_is_in_cascade_path(board, opponent_stacks, landing_coord))

def create_triangle_attack (victim_coords: list[Coord], attacker_coords: list[Coord], landing_coord: Coord) -> int:
    created_num = 0

    # Step 1: Find the existing stack in the attacker coordinate set that can possibly create the attack
    for coord_attacker in attacker_coords:
        # Suitable condition 1: in the same line
        # Suitable condition 2: distance = 2
        if get_Manhattan_distance(coord_attacker, landing_coord) == 2:
            # C1: Found in the same row, check whether there is a victim in between up or down
            if coord_attacker.r == landing_coord.r:
                desired_victim_c = max(coord_attacker.c, landing_coord.c) - 1
                desired_victim_r_up = landing_coord.r - 1
                desired_victim_r_down = landing_coord.r + 1
                
                if 0 <= desired_victim_r_up < BOARD_N and 0 <= desired_victim_c < BOARD_N:
                    desired_victim_coord_up = Coord(desired_victim_r_up, desired_victim_c)
                    if desired_victim_coord_up in victim_coords:
                        created_num += 1
                if 0 <= desired_victim_r_down < BOARD_N and 0 <= desired_victim_c < BOARD_N:
                    desired_victim_coord_down = Coord(desired_victim_r_down, desired_victim_c)
                    if desired_victim_coord_down in victim_coords:
                        created_num += 1
            
            # C2: Found in the same column, check whether there is a victim in between left or right
            elif coord_attacker.c == landing_coord.c:
                desired_victim_r = max(coord_attacker.r, landing_coord.r) - 1
                desired_victim_c_l = landing_coord.c - 1
                desired_victim_c_ri = landing_coord.c + 1
                
                if 0 <= desired_victim_r < BOARD_N and 0 <= desired_victim_c_l < BOARD_N:
                    desired_victim_coord_l = Coord(desired_victim_r, desired_victim_c_l)
                    if desired_victim_coord_l in victim_coords:
                        created_num += 1

                if 0 <= desired_victim_r < BOARD_N and 0 <= desired_victim_c_ri < BOARD_N:
                    desired_victim_coord_ri = Coord(desired_victim_r, desired_victim_c_ri)
                    if desired_victim_coord_ri in victim_coords:
                        created_num += 1

    return created_num

# Powerful form 1: same color triangle
def is_pwf_triangle (coords: list[Coord]) -> tuple[str, int]:
    # Further: convert the coords into a set
    explored_pair = set()
    triangles = set()

    for coord1 in coords:
        for coord2 in coords:
            if coord2 == coord1:
                continue

            pair_key = frozenset((coord1, coord2))
            if pair_key in explored_pair:
                continue

            explored_pair.add(pair_key)
            # Suitable condition 1: in the same line
            # Suitable condition 2: distance = 2
            if get_Manhattan_distance(coord1, coord2) == 2:
                # C1: Found in the same row, check whether there is an opponent in between up or down
                if coord1.r == coord2.r:
                    desired_c = max(coord1.c, coord2.c) - 1
                
                    for desired_r in [coord1.r - 1, coord1.r + 1]:
                        if 0 <= desired_r < BOARD_N and 0 <= desired_c < BOARD_N:
                            desired_coord = Coord(desired_r, desired_c)
                            if desired_coord in coords:
                                triangle_key = frozenset((coord1, coord2, desired_coord))
                                triangles.add(triangle_key)
                # C1: Found in the same column, check whether there is an opponent in between left or right
                elif coord1.c == coord2.c:
                    desired_r = max(coord1.r, coord2.r) - 1

                    for desired_c in [coord1.c - 1, coord1.c + 1]:
                        if 0 <= desired_r < BOARD_N and 0 <= desired_c < BOARD_N:
                            desired_coord = Coord(desired_r, desired_c)
                            if desired_coord in coords:
                                triangle_key = frozenset((coord1, coord2, desired_coord))
                                triangles.add(triangle_key)
    
    return ("triangle", len(triangles))

# Powerful form 2: same color adjacency
def is_pwf_adj (coords: list[Coord]) -> tuple[str, int]:
    explored_pair = set()
    num = 0

    for coord1 in coords:
        for coord2 in coords:
            if coord2 == coord1:
                continue

            pair_key = frozenset((coord1, coord2))
            if pair_key in explored_pair:
                continue

            explored_pair.add(pair_key)

            if is_in_same_line(coord1, coord2):
                num += 1
    
    return ("adjacency", num)


# Powerful form 3: same color step
def is_pwf_step (coords: list[Coord]) -> tuple[str, int]:
    explored_pair = set()
    steps = set()

    for coord1 in coords:
        for coord2 in coords:
            if coord2 == coord1:
                continue

            pair_key = frozenset((coord1, coord2))
            if pair_key in explored_pair:
                continue

            explored_pair.add(pair_key)

            # Suitable condition 1: There is a pair adjacent to each other--in the same row
            # Only check vertically, since the four orientations all contain the vertical base
            if (not is_adjacent(coord1, coord2)) or (coord1.r != coord2.r):
                continue

            # Suitable condition 2: There is a stack up or down
            desired_r_up = coord2.r - 1
            desired_r_down = coord2.r + 1
            desired_rs = [desired_r_up, desired_r_down]

            desired_c_l = min(coord1.c, coord2.c)
            desired_c_ri = max(coord1.c, coord2.c)
            desired_cs = [desired_c_l, desired_c_ri]

            for desired_r in desired_rs:
                for desired_c in desired_cs:
                    if not (0 <= desired_r < BOARD_N and 0 <= desired_c < BOARD_N):
                        continue

                    desired_coord = Coord (desired_r, desired_c)
                    if desired_coord in coords:
                        step_key = frozenset((coord1, coord2, desired_coord))
                        steps.add(step_key)

    return ("step", len(steps))



# Score 1: Whether placing our new stack on the specific coordinate can provide a triangle attack to the opponent stack

# Score 2: Whether placing our new stack on the specific coordinate can provide any powerful pattern
def pattern_set(pwf_results: list[tuple[str, int]]) -> set[str]:
    patterns = set()

    for pattern_name, count in pwf_results:
        if count > 0:
            patterns.add(pattern_name)

    return patterns

def get_create_pattern_score (opponent_coords: list[Coord], my_coords: list[Coord], landing_coord: Coord) -> float:
    # We need to evaluate through comparing to the previous board state
    prev_pwf_1 = is_pwf_triangle(my_coords)
    prev_pwf_2 = is_pwf_adj(my_coords)
    prev_pwf_3 = is_pwf_step(my_coords)
    prev_score = prev_pwf_1[1] + prev_pwf_2[1] + prev_pwf_3[1]
    prev_patterns = pattern_set([prev_pwf_1, prev_pwf_2, prev_pwf_3])

    coords_added = my_coords + [landing_coord]
    new_pwf_1 = is_pwf_triangle(coords_added)
    new_pwf_2 = is_pwf_adj(coords_added)
    new_pwf_3 = is_pwf_step(coords_added)
    new_score = new_pwf_1[1] + new_pwf_2[1] + new_pwf_3[1]
    new_patterns = pattern_set([new_pwf_1, new_pwf_2, new_pwf_3])

    # No new patterns by landing the stack on the specific coordinate
    if new_score <= prev_score:
        return 0.0
    
    # There would appear new pattern
    total_pattern_score = new_score - prev_score
    # Special case: if the new placement can create a pattern that the opponent already has but we did not have previously, we should score it high
    # Get what the opponent already has
    opponent_pwf_1 = is_pwf_triangle(opponent_coords)
    opponent_pwf_2 = is_pwf_adj(opponent_coords)
    opponent_pwf_3 = is_pwf_step(opponent_coords)

    opponent_patterns = pattern_set([opponent_pwf_1, opponent_pwf_2, opponent_pwf_3])
    for pattern in opponent_patterns:
        if (pattern not in prev_patterns) and (pattern in new_patterns):
            total_pattern_score *= 2
            break
    
    return float(total_pattern_score)

# Score 3: Whether placing our new stack on the specific coordinate can prevent the opponent from building a triangle attack to our stack in the next step
def get_protection_score (opponent_coords: list[Coord], my_coords: list[Coord], landing_coord: Coord) -> int:
    # Switch my with opponent
    # If the opponent put a new stack at the specific coordinate, would it be able to create a triangle attack to our existing stacks
    return create_triangle_attack(my_coords, opponent_coords, landing_coord)

# Socre 4: Whether in the same line with any opponent stacks, even better can perform cascade
def get_same_line_score (board: dict[Coord, CellState], opponent_coords: list[Coord], landing_coord: Coord, landing_state: CellState) -> float:
    score = 0

    # Whether can fall on the same line
    for coord_opponent in opponent_coords:
        if is_in_same_line(landing_coord, coord_opponent):
            score += 1
            attack_direction = get_same_direction(landing_coord, coord_opponent)
            if attack_direction is not None and successful_cascade(board, landing_coord, landing_state, coord_opponent, attack_direction):
                score += 1  # might need adjustion
    
    return score

# Score 5: The distance to the nearest enemy stack
def get_nearest_enemy_distance (opponent_coords: list[Coord], landing_coord: Coord) -> float:
    if not opponent_coords:
        return 0.0

    nearest = min(
        get_Manhattan_distance(landing_coord, coord)
        for coord in opponent_coords
    )

    if nearest > 0:
        return 1 / nearest

    return 1.0

# Detect board state for weight adjustment and special cases
def has_edge_pressure (coords: list[Coord]) -> bool:
    edge_options = []

    for coord in coords:
        labels = get_edge_labels(coord)
        if labels:
            edge_options.append(labels)

    if len(edge_options) < 3:
        return False

    for assignment in product(*edge_options):
        if len(set(assignment)) >= 3:
            return True

    return False

def has_corner_pressure_get_remain_corner(
    coords: list[Coord]
) -> tuple[str, Coord] | None:
    corners = {
        "top_left": Coord(0, 0),
        "top_right": Coord(0, BOARD_N - 1),
        "bottom_left": Coord(BOARD_N - 1, 0),
        "bottom_right": Coord(BOARD_N - 1, BOARD_N - 1),
    }

    occupied_corners = set()

    for coord in coords:
        corner = get_corner_label(coord)
        if corner is not None:
            occupied_corners.add(corner)

    if len(occupied_corners) < 3:
        return None

    remaining_labels = set(corners.keys()) - occupied_corners

    # Case 1: exactly 3 corners occupied, so 1 remaining corner
    if remaining_labels:
        label = next(iter(remaining_labels))
        return label, corners[label]

    # Case 2: all 4 corners occupied, choose any random corner
    label = random.choice(list(corners.keys()))
    return label, corners[label]

def is_fortressed (coords: list[Coord]) -> bool:
    pairwise_dists = []
    for coord1 in coords:
        for coord2 in coords:
            pairwise_dists.append(get_Manhattan_distance(coord1, coord2))
    
    return max(pairwise_dists) <= 2
