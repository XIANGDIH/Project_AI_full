# This file contains the logic about finding "EVAL"--the heuristic scoring for the placement phase: 
# For the placement phase, we play by pure logic;
# For the play phase, we build the evaluation function with features and weights.


import random

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, INITIAL_STACK_HEIGHT, BOARD_N, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .helper_placement import in_safe_area, in_possible_area, create_triangle_attack, get_create_pattern_score, \
    get_protection_score, get_same_line_score, get_nearest_enemy_distance, is_fortressed, has_edge_pressure, has_corner_pressure_get_remain_corner
from .helper import get_closest_to_centre


# ----------------------------
# Placement phase (first 8 turns in total, 4 per player)
# ----------------------------

def choose_coord_placement_phase(
    board: dict[Coord, CellState],
    legal_actions: list[Action],
    color: PlayerColor,
    total_turn_count: int) -> Coord:
    """
    Generate all legal and preferred coordinates for the current agent from self._board.
    This returns a (random) preferred (trying to be optimal) coordinate where we should
    place our next stack on the board.
    """
    centre_coords: list[Coord] = [Coord(3, 3), Coord(3, 4), Coord(4, 3), Coord(4, 4)]
    legal_coords: list[Coord] = []

    # Get the current placed stacks on the current board situation
    blue_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.BLUE]
    blue_coords = [c for c, s in board.items() if s.color == PlayerColor.BLUE]
    red_stacks = [(c, s) for c, s in board.items() if s.color == PlayerColor.RED]
    red_coords = [c for c, s in board.items() if s.color == PlayerColor.RED]

    if color == PlayerColor.RED:
        my_stacks = red_stacks
        my_coords = red_coords
        opponent_stacks = blue_stacks
        opponent_coords = blue_coords
    else:
        my_stacks = blue_stacks
        my_coords = blue_coords
        opponent_stacks = red_stacks
        opponent_coords = red_coords

    # We should only consider the legal coordinates--empty coordinates that are legal according to the game rules
    legal_coords = [action.coord for action in legal_actions]
    empty_coords = legal_coords

    # For the first turn
    if color == PlayerColor.RED and total_turn_count == 0:
        return random.choice(centre_coords)
    # If we are Blue
    if color == PlayerColor.BLUE and total_turn_count == 0:
        # Get the placed Red stack (placed by the enemy)
        placed_enemy_coord = red_coords[0]

        # Check whether this placed stacks has taken one of the centre coordinates
        # Condition 1: The enemy has not placed in the centre yet
        if placed_enemy_coord not in centre_coords:
            return random.choice(centre_coords)
            
        # Condition 2: The enemy has placed in the centre, then only one of the centre coordinates could be taken by us
        # This remaining legal coordinate should in both the centre coodinates list and the empty list we have found before (or the distance to the placed enemy stack equals to 2)
        for coord_cr in centre_coords:
            # Alternative: get_Manhattan_distance(coord_cr, placed_enemy_coord) == 2
            if coord_cr in empty_coords:
                return coord_cr
    
    # For the second turn
    if total_turn_count == 1:
        centre_taken = False    # A flag for checking whether the centre is full now
        placed_coord_by_us = None
        
        for centre_coord in centre_coords:
            if centre_coord in board.keys():
                # If the oppoenent also takes one of the centre coordinates in their turn--the four centre coordinates cannot be placed with any new stacks from now on
                if board[centre_coord].color != color:
                    centre_taken = True
                # If there's still legal centre coordinates (the opponents still do not take the centre coordinates) 
                else :
                    placed_coord_by_us = centre_coord

        if centre_taken == False:
            # Place the our new stack right beside the existing stack at the centre--in the same row or column, both are fine
            for centre_coord in centre_coords:
                if centre_coord in empty_coords and (
                    (centre_coord.c == placed_coord_by_us.c and centre_coord.r != placed_coord_by_us.r)
                    or
                    (centre_coord.r == placed_coord_by_us.r and centre_coord.c != placed_coord_by_us.c)
                ):
                    return centre_coord
        else:
            ## Search in the safe area
            # Step 1: Get coordinates in the safe area
            safe_area_coords = get_safe_area(board, empty_coords, opponent_stacks)
            # Step 2: Get a random best coordinate with greatest evaluation score
            best = get_best_coordinate(board, safe_area_coords, opponent_coords, my_coords, color)

            # Defensive fallback
            if best is not None:
                print("Testing: Correct in turn 2. \n")
                return best

            return random.choice(legal_coords)
            
    # For the third and forth turn
    # Special case: Corner pressure
    operational_coord = has_corner_pressure_get_remain_corner(opponent_coords)
    is_my_last_placement = total_turn_count == 3

    if is_my_last_placement and operational_coord is not None:
        match operational_coord[0]:
            case "top_left":
                if Coord(1, 1) not in empty_coords:
                    return Coord(1, 2)
                return Coord(1, 1)
            case "top_right":
                if Coord(1, BOARD_N - 2) not in empty_coords:
                    return Coord(1, BOARD_N - 3)
                return Coord(1, BOARD_N - 2)
            case "bottom_left":
                if Coord(BOARD_N - 2, 1) not in empty_coords:
                    return Coord(BOARD_N - 2, 2)
                return Coord(BOARD_N - 2, 1)
            case "bottom_right":
                if Coord(BOARD_N - 2, BOARD_N - 2) not in empty_coords:
                    return Coord(BOARD_N - 2, BOARD_N - 3)
                return Coord(BOARD_N - 2, BOARD_N - 2)

    # Step 1: Get coordinates in the safe area
    safe_area_coords = get_safe_area(board, empty_coords, opponent_stacks)

    if len(safe_area_coords) == 0:
        # Expand the searching area to all possible coordinates expect those could be directly pushed off by cascade of the enemy stacks
        possible_area_coords = get_possible_area(board, empty_coords, opponent_stacks)
        best = get_best_coordinate(board, possible_area_coords, opponent_coords, my_coords, color)

        # Defensive fallback
        if best is not None:
            print("Testing: Correct in turn 3-extended. \n")
            return best

        return random.choice(legal_coords)


    # Step 2: Get the best coordinate which is closest to the centre of the board with greatest evaluation score
    best = get_best_coordinate(board, safe_area_coords, opponent_coords, my_coords, color)

    # Defensive fallback
    if best is not None:
        print("Testing: Correct in turn 3. \n")
        return best

    return random.choice(legal_coords)

# NOT in use yet
def get_empty_coords (opponent_coords: list[Coord], my_coords: list[Coord]) -> list[Coord]:
    empty_coord = []
    # Row
    for i in range(8):
        # Column
        for j in range(8):
            coord_current = Coord(i, j)
            if (coord_current not in opponent_coords) and (coord_current not in my_coords):
                empty_coord.append(coord_current)
    
    return empty_coord

def get_safe_area (board: dict[Coord, CellState], empty_coords: list[Coord], opponent_stacks: list[tuple[Coord, CellState]]) -> list[Coord]:
    safe_coords: list[Coord] = []

    for empty_coord in empty_coords:
        if in_safe_area(board, opponent_stacks, empty_coord):
            safe_coords.append(empty_coord)

    return safe_coords

def get_possible_area (board: dict[Coord, CellState], empty_coords: list[Coord], opponent_stacks: list[(Coord, CellState)]) -> list[Coord]:
    possible_coords: list[Coord] = []

    for empty_coord in empty_coords:
        if in_possible_area(board, opponent_stacks, empty_coord):
            possible_coords.append(empty_coord)

    return possible_coords

def get_best_coordinate (board: dict[Coord, CellState], possible_coords: list[Coord], opponent_coords: list[Coord], my_coords: list[Coord], color: PlayerColor) -> Coord:
    # A dictionary to store the coordinate and its corresponding score under consideration
    score_dict = {}
    # Set the cell state as our color with height of three
    landing_state = CellState(color, 3)

    # Step 1: Obtain the "evaluation" score for each safe and empty coordinate in the current board
    for landing_coord in possible_coords:
        score_dict[landing_coord] = get_coord_score(board, opponent_coords, my_coords, landing_coord, landing_state)
            
    # Step 2: Get the coordinates with the greatest score
    # Defensive check needed???
    max_score = max(score_dict.values())
    best_coordinates = [k for k, v in score_dict.items() if v == max_score]
    # Pick one coordinate randomly: random.choice(best_coordinates)
    # Pick the one closest to the centre of the board
    return get_closest_to_centre(best_coordinates)

def get_coord_score (board: dict[Coord, CellState], opponent_coords: list[Coord], my_coords: list[Coord], landing_coord: Coord, landing_state: CellState) -> float:
    total_score = 0.0

    weight_triangle_attack = 1.0
    weight_pattern = 1.0
    weight_same_line = 1.0
    weight_protection = 1.0
    weight_nearest_enemy_distance = 1.0

    # Adjust the weight according to different board state, when the number of enemy stacks > 2
    if len(opponent_coords) > 2:
        if has_edge_pressure(opponent_coords):
            weight_triangle_attack = 0.2
            weight_pattern = 0.2
            weight_same_line = 2.0
            weight_protection = 0.2
            weight_nearest_enemy_distance = 1.2
        elif is_fortressed(opponent_coords):
            weight_triangle_attack = 1.5
            weight_pattern = 1.2
            weight_same_line = 0.3
            weight_protection = 1.5
            weight_nearest_enemy_distance = 0.3

    triangle_attack_score = create_triangle_attack(opponent_coords, my_coords, landing_coord)
    pattern_score  = get_create_pattern_score(opponent_coords, my_coords, landing_coord)
    protection_score  = get_protection_score(opponent_coords, my_coords, landing_coord)
    same_line_score = get_same_line_score(board, opponent_coords, landing_coord, landing_state)
    nearest_enemy_score = get_nearest_enemy_distance(opponent_coords, landing_coord)

    total_score = (
        weight_triangle_attack * triangle_attack_score
        + weight_pattern * pattern_score
        + weight_protection * protection_score
        + weight_same_line * same_line_score
        + weight_nearest_enemy_distance * nearest_enemy_score
    )

    return total_score
    
                    
