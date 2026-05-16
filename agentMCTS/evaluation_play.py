# This file contains the logic about finding "EVAL" for the play phase.
# We reuse old heuristic ideas, but adapt them for two-player evaluation.


from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, CellState, BOARD_N, MoveAction

from .helper import get_same_direction, successful_cascade, is_adjacent, get_opposite_direction, is_in_same_line, encode_state, get_Manhattan_distance, meaningful_cascade, copy_state
from .helper_play import BoardState, detect_board_state, get_threat, get_total_dist_to_edge, has_immediate_elimination
from .rules import get_legal_actions, apply_action
from .types import SeenStates


# ----------------------------
# Main eval for play phase
# ----------------------------

def evaluate (
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int
) -> float:
    """
    Bigger score = better board for this player.
    """
    player_stacks = [(c, s) for c, s in board.items() if s.color == color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == color.opponent]

    # Quick win/lose checks
    if not opponent_stacks:
        return 1000000.0
    if not player_stacks:
        return -1000000.0

    dist_weight = 0.1
    threat_weight = 0.5
    opponent_escapability_weight = 0.05

    state = detect_board_state(board, opponent_stacks, player_stacks)
    if BoardState.COMPACT_ALIGNMENT in state:
        if BoardState.PLAYER_SCARCITY in state:
            dist_weight -= 0.06
            threat_weight += 0.065
        else:
            dist_weight -= 0.05
            threat_weight += 0.06

    elif BoardState.OPPONENT_SCATTERED in state:
        if BoardState.PLAYER_SCARCITY in state:
            dist_weight += 0.02
            threat_weight += 0.01
        else:
            dist_weight += 0.03
            threat_weight -= 0.04

    elif BoardState.PLAYER_SCARCITY in state:
        dist_weight -= 0.02
        threat_weight += 0.04

    if BoardState.IN_OUR_FAVOUR in state or BoardState.BALANCED in state:
        opponent_escapability_weight += 1.0

    if len(opponent_stacks) == 1:
        opponent_escapability_weight += 1.5

    total_dist = 0.0
    total_threat = 0.0
    for coord_opponent, state_opponent in opponent_stacks:
        best_dist = float("inf")
        best_threat = float("inf")

        for coord_player, state_player in player_stacks:
            # Distance
            d = abs(coord_opponent.r - coord_player.r) + abs(coord_opponent.c - coord_player.c)
            best_dist = min(best_dist, d)
            # Threat
            t = get_threat(coord_player, state_player, coord_opponent, state_opponent, board, state)
            best_threat = min(best_threat, t)

        total_dist += best_dist
        total_threat += best_threat

    safe_action_num = get_f9_score_fast(board, color, opponent_stacks, player_stacks)

    # Old heuristic was "lower is better"; we flip it to "higher is better".
    return -(
        len(opponent_stacks)
        + dist_weight * total_dist
        + threat_weight * total_threat
        + opponent_escapability_weight * safe_action_num
    )


# {Features}
# Feature 1: Difference in the stack number on the board
def get_f1_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    return 12 - len(opponent_stacks)

# Feature 2: Difference in the total stack height on the board
def get_f2_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    # Get the total height of stacks for our player
    total_height_player = 0.0
    for _, state_player in player_stacks:
        total_height_player += state_player.height

    # Get the total height of stacks for the opponent
    total_height_opponent = 0.0
    for _, state_opponent in opponent_stacks:
        total_height_opponent += state_opponent.height
    
    return total_height_player - total_height_opponent
# Feature 2_new: Difference in the greatest stack height on the board
def get_f2_score_new (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    # Get the greatest height of stacks for our player
    greatest_height_player = 0.0
    for _, state_player in player_stacks:
        if state_player.height > greatest_height_player:
            greatest_height_player = state_player.height

    # Get the greatest height of stacks for the opponent
    greatest_height_opponent = 0.0
    for _, state_opponent in opponent_stacks:
        if state_opponent.height > greatest_height_opponent:
            greatest_height_opponent = state_opponent.height
    
    return greatest_height_player - greatest_height_opponent

# Feature 3: Difference in the total legal actions that could be considered for the current board
# This might be expensive
def get_f3_score (board: dict[Coord, CellState], color_player: PlayerColor, total_turn_count: int) -> float:
    # Find the opponent's color
    color_opponent = None
    if color_player == PlayerColor.RED:
        color_opponent = PlayerColor.BLUE
    else:
        color_opponent = PlayerColor.RED
    
    # Get legal actions for our player
    actions_player = get_legal_actions(board, color_player, total_turn_count)
    
    # Get legal actionf for the opponent player
    actions_opponent = get_legal_actions(board, color_opponent, total_turn_count)

    return len(actions_player) - len(actions_opponent)

# Feature 4: Difference in the direct EAT for the current board
def get_f4_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_eat = []
    player_eat = []

    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            # Check whether they are adjacent
            if not is_adjacent(coord_opponent, coord_player):
                continue
            else:
                # They are adjacent, compare their height
                # Since the next turn is my opponent's turn--if the height is similar, it should be considered as the opponent's strength--?
                # BUT the opponent may not move this... "=" should be counted for both sides
                if state_opponent.height <= state_player.height:
                    player_eat.append((coord_player, state_player))
                elif state_opponent.height >= state_player.height:
                    opponent_eat.append((coord_opponent, state_opponent))
    
    return len(player_eat) - len(opponent_eat)

# Feature 5: Difference in the direct/possible future Casecade count for the current board
def get_f5_score (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_cascade = []
    player_cascade = []
    opponent_cascade_m = []
    player_cascade_m = []

    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            cascade_direction_player = get_same_direction(coord_player, coord_opponent)

            # Check whether they are on the same line
            if cascade_direction_player == None:
                continue

            # There are on the same line, and we get the direction for regarding our player as the attacker first
            # Get the opposite direction
            cascade_direction_opponent = get_opposite_direction(cascade_direction_player)
            # Check whether our player can play a successful cascade
            if successful_cascade(board, coord_player, state_player, coord_opponent, cascade_direction_player):
                player_cascade.append((coord_player, state_player))
            elif meaningful_cascade(coord_player, state_player, opponent_stacks, cascade_direction_player):
                player_cascade_m.append((coord_player, state_player))
            # Check whether the opponent can play a successful cascade
            if successful_cascade(board, coord_opponent, state_opponent, coord_player, cascade_direction_opponent):
                opponent_cascade.append((coord_opponent, state_opponent))
            elif meaningful_cascade(coord_opponent, state_opponent, player_stacks, cascade_direction_opponent):
                opponent_cascade_m.append((coord_opponent, state_opponent))
                
    direct_cascade_diff = len(player_cascade) - len(opponent_cascade)
    indirect_cascade_diff = len(player_cascade_m) - len(opponent_cascade_m)

    return 2.0 * direct_cascade_diff + 0.5 * indirect_cascade_diff

# Feature 6: Difference in the meaningful same-line count for the current board--this is use as an additional feature
def get_f6_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_same_line = []
    player_same_line = []

    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            # Check whether they are on the same line
            is_same_line = is_in_same_line(coord_player, coord_opponent)
            if not is_same_line:
                continue

            # Since the next turn is my opponent's turn--if the height is similar, it should be considered as the opponent's strength--Might not be preferable
            if state_player.height >= state_opponent.height:
                player_same_line.append((coord_player, state_player))
            elif state_player.height <= state_opponent.height:
                opponent_same_line.append((coord_opponent, state_opponent))
    
    return len(player_same_line) - len(opponent_same_line)

# Feature 7: Difference in average safety distance from nearest edge
def get_f7_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    player_total_dist = get_total_dist_to_edge(player_stacks)
    opponent_total_dist = get_total_dist_to_edge(opponent_stacks)

    player_average = player_total_dist / len(player_stacks)
    opponent_average = opponent_total_dist / len(opponent_stacks)

    return player_average - opponent_average

# Feature 8: Penalty for existing state
def get_f8_score (board: dict[Coord, CellState],seen_states: SeenStates
) -> float:
    encoded = encode_state(board)

    seen_count = seen_states.get(encoded, (0, None))[0]

    if seen_count == 1:
        return -150.0
    elif seen_count == 2:
        return -155.0
    elif seen_count >= 3:
        return -100000.0

    return 0.0

# Feature 9: The escapability of the opponent stacks
def get_f9_score (board, my_color, total_turn_count):
    opponent = my_color.opponent
    safe_count = 0

    opponent_actions = get_legal_actions(board, opponent, total_turn_count)

    for opp_action in opponent_actions:
        # Only count actual escape movement
        if not isinstance(opp_action, MoveAction):
            continue

        next_board = copy_state(board)
        apply_action(next_board, opponent, opp_action)

        my_actions = get_legal_actions(next_board, my_color, total_turn_count + 1)

        can_punish = False
        for my_action in my_actions:
            punish_board = copy_state(next_board)
            apply_action(punish_board, my_color, my_action)

            opponent_still_alive = any(
                cell.color == opponent
                for cell in punish_board.values()
            )

            if not opponent_still_alive:
                can_punish = True
                break

        if not can_punish:
            safe_count += 1

    return safe_count

def get_f9_score_fast (
    board: dict[Coord, CellState],
    my_color: PlayerColor,
    opponent_stacks: list[tuple[Coord, CellState]],
    player_stacks: list[tuple[Coord, CellState]]
) -> int:
    safe_count = 0

    for coord_opponent, state_opponent in opponent_stacks:
        for direction in CARDINAL_DIRECTIONS:
            target_r = coord_opponent.r + direction.r
            target_c = coord_opponent.c + direction.c

            if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                continue

            target_coord = Coord(target_r, target_c)
            target_state = board.get(target_coord)
            if target_state is not None and target_state.color != my_color.opponent:
                continue

            is_immediately_punishable = False
            for coord_player, state_player in player_stacks:
                if is_adjacent(coord_player, target_coord) and state_player.height >= state_opponent.height:
                    is_immediately_punishable = True
                    break

            if not is_immediately_punishable:
                safe_count += 1

    return safe_count
    
# Feature 10: Closest distance to the future attackable stack
def get_f10_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    shortest_dist_player = 10.0
    shortest_dist_opponent = 10.0
    for coord_player, state_player in player_stacks:
        for coord_opponent, state_opponent in opponent_stacks:
            distance = get_Manhattan_distance(coord_player, coord_opponent)
            if distance < shortest_dist_player and state_player.height >= state_opponent.height:
                shortest_dist_player = distance
            if distance < shortest_dist_opponent and state_opponent.height >= state_player.height:
                shortest_dist_opponent = distance
    
    return -shortest_dist_player
    
def evaluate_new (
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int,
    seen_states: SeenStates,
    root_board_state: list[BoardState]
) -> float:
    """
    Bigger score = better board for this player.
    """
    player_stacks = [(c, s) for c, s in board.items() if s.color == color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == color.opponent]

    # Quick win/lose checks
    if not opponent_stacks:
        #print("DEBUG: Condition Valid\n")
        return 1000000.0
    if not player_stacks:
        return -1000000.0

    # The original weights
    f1_weight = 40.0
    f2_weight = 5.0
    f3_weight = 0.5
    f4_weight = 5.0
    f5_weight = 5.0
    f6_weight = 1.0
    f7_weight = 2.0
    f9_weight = 0.0
    f10_weight = 3.0

    # Adjust weights based on current board pattern--need to be updated
    #state = detect_board_state(board, opponent_stacks, player_stacks)
    state = root_board_state
    if BoardState.COMPACT_ALIGNMENT in state:
        if BoardState.PLAYER_SCARCITY in state:
            f1_weight += 0.3 * 20
            f2_weight += 0.3 * 1
        else:
            f2_weight += 0.2 * 1

        if BoardState.EDGE_CORNER_PRESSURE in state:
            f5_weight += 3.0
            f7_weight += 0.2 * 2

    elif BoardState.OPPONENT_SCATTERED in state:
        if BoardState.PLAYER_SCARCITY in state:
            f1_weight += 0.2 * 20
            f2_weight += 0.1 * 1
            f6_weight += 0.3 * 1
        else:
            f6_weight += 0.4 * 1

        if BoardState.EDGE_CORNER_PRESSURE in state:
            f5_weight += 5.0
            f7_weight += 0.3 * 2

    if BoardState.PLAYER_SCARCITY in state:
        f1_weight += 0.3 * 20
        f2_weight += 0.2 * 1
    elif BoardState.IN_OUR_FAVOUR in state and BoardState.HAS_IMMEDIATE_AFFECT not in state:
        f6_weight += 5.0
        f10_weight += 10.0
    
    if BoardState.EDGE_CORNER_PRESSURE in state:
        f5_weight += 3.0
        f7_weight += 0.2 * 2
    
    if BoardState.ATTACKABLE_OPPONENT_FAR in state:
        f10_weight += 10.0
        f6_weight += 5.0

    if BoardState.HAS_IMMEDIATE_AFFECT in state:
        f1_weight += 0.3 * 20

    if BoardState.IN_OUR_FAVOUR in state or BoardState.BALANCED in state:
        f9_weight += 15.0

    # Get the scores for eac feature
    feature1_stack_num_diff = get_f1_score(opponent_stacks, player_stacks)
    feature2_stack_height_diff = get_f2_score(opponent_stacks,player_stacks)
    feature3_legal_action_diff = get_f3_score(board, color, total_turn_count)
    feature4_eat_diff = get_f4_score(opponent_stacks, player_stacks)
    feature5_cascade_diff = get_f5_score(board, opponent_stacks, player_stacks)
    feature6_same_line_diff = get_f6_score(opponent_stacks, player_stacks)
    feature7_average_edge_dist_diff = get_f7_score(opponent_stacks, player_stacks)
    feature8_has_seen_penalty = get_f8_score(board, seen_states)
    feature9_opponent_escapability = get_f9_score(board, color, total_turn_count)
    feature10_attackable_closest_dist = get_f10_score(opponent_stacks, player_stacks)

    return (
        + f1_weight * feature1_stack_num_diff
        + f2_weight * feature2_stack_height_diff
        #+ f3_weight * feature3_legal_action_diff
        + f4_weight * feature4_eat_diff
        + f5_weight * feature5_cascade_diff
        + f6_weight * feature6_same_line_diff
        #+ f7_weight * feature7_average_edge_dist_diff
        + feature8_has_seen_penalty
        - f9_weight * feature9_opponent_escapability
        + f10_weight * feature10_attackable_closest_dist
    )
