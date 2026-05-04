# This file contains the logic about finding "EVAL" for the play phase.
# We reuse old heuristic ideas, but adapt them for two-player evaluation.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N

from .helper import get_same_direction, successful_cascade, is_adjacent, get_opposite_direction, is_in_same_line
from .helper_play import BoardState, detect_board_state, get_threat, get_total_dist_to_edge
from .rules import get_legal_actions


# ----------------------------
# Main eval for play phase
# ----------------------------

def evaluate (
    board: dict[Coord, CellState],
    color: PlayerColor
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

    # Adjust weights based on current board pattern
    state = detect_board_state(opponent_stacks, player_stacks)
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

    # Old heuristic was "lower is better"; we flip it to "higher is better".
    return -(
        len(opponent_stacks)
        + dist_weight * total_dist
        + threat_weight * total_threat
    )


# {Features}
# Feature 1: Difference in the stack number on the board
def get_f1_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    return len(player_stacks) - len(opponent_stacks)

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
                if state_opponent.height < state_player.height:
                    player_eat.append((coord_player, state_player))
                else:
                    opponent_eat.append((coord_opponent, state_opponent))
    
    return len(player_eat) - len(opponent_eat)

# Feature 5: Difference in the direct Casecade count for the current board
def get_f5_score (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_cascade = []
    player_cascade = []


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
            # Check whether the opponent can play a successful cascade
            if successful_cascade(board, coord_opponent, state_opponent, coord_player, cascade_direction_opponent):
                opponent_cascade.append((coord_opponent, state_opponent))
    
    return len(player_cascade) - len(opponent_cascade)

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

            # Since the next turn is my opponent's turn--if the height is similar, it should be considered as the opponent's strength--?
            if state_player.height > state_opponent.height:
                player_same_line.append((coord_player, state_player))
            else:
                opponent_same_line.append((coord_opponent, state_opponent))
    
    return len(player_same_line) - len(opponent_same_line)

# Feature 7: Difference in average safety distance from nearest edge
def get_f7_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    player_total_dist = get_total_dist_to_edge(player_stacks)
    opponent_total_dist = get_total_dist_to_edge(opponent_stacks)

    player_average = player_total_dist / len(player_stacks)
    opponent_average = opponent_total_dist / len(opponent_stacks)

    return player_average - opponent_average
    
def evaluate_new (
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

    # The original weights
    f1_weight = 20.0
    f2_weight = 5.0
    f3_weight = 0.5
    f4_weight = 15.0
    f5_weight = 10.0
    f6_weight = 1.0
    f7_weight = 2.0

    # Adjust weights based on current board pattern--need to be updated
    state = detect_board_state(opponent_stacks, player_stacks)
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
    if BoardState.EDGE_CORNER_PRESSURE in state:
            f5_weight += 3.0
            f7_weight += 0.2 * 2

    # Get the scores for eac feature
    feature1_stack_num_diff = get_f1_score(opponent_stacks, player_stacks)
    feature2_stack_height_diff = get_f2_score(opponent_stacks,player_stacks)
    feature3_legal_action_diff = get_f3_score(board, color, total_turn_count)
    feature4_eat_diff = get_f4_score(opponent_stacks, player_stacks)
    feature5_cascade_diff = get_f5_score(board, opponent_stacks, player_stacks)
    feature6_same_line_diff = get_f6_score(opponent_stacks, player_stacks)
    feature7_average_edge_dist_diff = get_f7_score(opponent_stacks, player_stacks)

    return (
        + f1_weight * feature1_stack_num_diff
        + f2_weight * feature2_stack_height_diff
        + f3_weight * feature3_legal_action_diff
        + f4_weight * feature4_eat_diff
        + f5_weight * feature5_cascade_diff
        + f6_weight * feature6_same_line_diff
        + f7_weight * feature7_average_edge_dist_diff
    )



