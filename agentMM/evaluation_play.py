# This file holds play-phase evaluation logic.
# It reuses old heuristic ideas in a two-player scoring style.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N

from .helper import get_same_direction, successful_cascade, is_adjacent, get_opposite_direction, is_in_same_line
from .helper_play import BoardState, detect_board_state, get_threat
from .rules import get_legal_actions


# ----------------------------
# Main play-phase eval
# ----------------------------

def evaluate (
    board: dict[Coord, CellState],
    color: PlayerColor
) -> float:
    """
    Higher score means a better board for this player.
    """
    player_stacks: list[tuple[Coord, CellState]] = []
    opponent_stacks: list[tuple[Coord, CellState]] = []

    for coord, cell_state in board.items():
        if cell_state.color == color:
            player_stacks.append((coord, cell_state))
        else:
            opponent_stacks.append((coord, cell_state))

    # Quick win/lose checks.
    if not opponent_stacks:
        return 1000000.0
    if not player_stacks:
        return -1000000.0

    dist_weight = 0.1
    threat_weight = 0.5

    # Tweak weights based on the current board pattern.
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
            # Distance term.
            d = abs(coord_opponent.r - coord_player.r) + abs(coord_opponent.c - coord_player.c)
            best_dist = min(best_dist, d)
            # Threat term.
            t = get_threat(coord_player, state_player, coord_opponent, state_opponent, board, state)
            best_threat = min(best_threat, t)

        total_dist += best_dist
        total_threat += best_threat

    # Old version was "lower is better", so we flip the sign here.
    return -(
        len(opponent_stacks)
        + dist_weight * total_dist
        + threat_weight * total_threat
    )


# Feature 1: stack count difference
def get_f1_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    return len(player_stacks) - len(opponent_stacks)

# Feature 2: total stack height difference
def get_f2_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    # Sum up our total stack height.
    total_height_player = 0.0
    for _, state_player in player_stacks:
        total_height_player += state_player.height

    # Sum up opponent total stack height.
    total_height_opponent = 0.0
    for _, state_opponent in opponent_stacks:
        total_height_opponent += state_opponent.height
    
    return total_height_player - total_height_opponent

# Feature 3: legal action count difference
# This one can be expensive.
def get_f3_score (board: dict[Coord, CellState], color_player: PlayerColor, total_turn_count: int) -> float:
    # Figure out opponent color.
    color_opponent = None
    if color_player == PlayerColor.RED:
        color_opponent = PlayerColor.BLUE
    else:
        color_opponent = PlayerColor.RED
    
    # Count legal actions for us.
    actions_player = get_legal_actions(board, color_player, total_turn_count)
    
    # Count legal actions for opponent.
    actions_opponent = get_legal_actions(board, color_opponent, total_turn_count)

    return len(actions_player) - len(actions_opponent)

# Feature 4: direct EAT chance difference
def get_f4_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_eat = []
    player_eat = []

    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            # Skip if they are not adjacent.
            if not is_adjacent(coord_opponent, coord_player):
                continue
            else:
                # They are adjacent, so compare heights.
                if state_opponent.height < state_player.height:
                    player_eat.append((coord_player, state_player))
                else:
                    opponent_eat.append((coord_opponent, state_opponent))
    
    return len(player_eat) - len(opponent_eat)

# Feature 5: direct CASCADE chance difference
def get_f5_score (board: dict[Coord, CellState], opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_cascade = []
    player_cascade = []


    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            cascade_direction_player = get_same_direction(coord_player, coord_opponent)

            # Must be on the same row or column.
            if cascade_direction_player == None:
                continue

            # Get the reverse direction for the opponent side.
            cascade_direction_opponent = get_opposite_direction(cascade_direction_player)
            # Check if our cascade can directly remove them.
            if successful_cascade(board, coord_player, state_player, coord_opponent, cascade_direction_player):
                player_cascade.append((coord_player, state_player))
            # Check if opponent can directly remove us.
            if successful_cascade(board, coord_opponent, state_opponent, coord_player, cascade_direction_opponent):
                opponent_cascade.append((coord_opponent, state_opponent))
    
    return len(player_cascade) - len(opponent_cascade)

# Feature 6: useful same-line pressure difference
def get_f6_score (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    opponent_same_line = []
    player_same_line = []

    for coord_opponent, state_opponent in opponent_stacks:
        for coord_player, state_player in player_stacks:
            # Only care about same row or column.
            is_same_line = is_in_same_line(coord_player, coord_opponent)
            if not is_same_line:
                continue

            # If our stack is taller, count it for us; otherwise for opponent.
            if state_player.height > state_opponent.height:
                player_same_line.append((coord_player, state_player))
            else:
                opponent_same_line.append((coord_opponent, state_opponent))

    return len(player_same_line) - len(opponent_same_line)


def get_feature_breakdown(
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int
) -> dict[str, float]:
    """
    Compute all feature values for the current board.
    We use this for logging and weight tuning.
    """
    player_stacks: list[tuple[Coord, CellState]] = []
    opponent_stacks: list[tuple[Coord, CellState]] = []

    for coord, cell_state in board.items():
        if cell_state.color == color:
            player_stacks.append((coord, cell_state))
        else:
            opponent_stacks.append((coord, cell_state))

    feature1_stack_num_diff = get_f1_score(opponent_stacks, player_stacks)
    feature2_stack_height_diff = get_f2_score(opponent_stacks, player_stacks)
    feature3_legal_action_diff = get_f3_score(board, color, total_turn_count)
    feature4_eat_diff = get_f4_score(opponent_stacks, player_stacks)
    feature5_cascade_diff = get_f5_score(board, opponent_stacks, player_stacks)
    feature6_same_line_diff = get_f6_score(opponent_stacks, player_stacks)

    feature_map = {
        "f1_stack_num_diff": feature1_stack_num_diff,
        "f2_stack_height_diff": feature2_stack_height_diff,
        "f3_legal_action_diff": feature3_legal_action_diff,
        "f4_eat_diff": feature4_eat_diff,
        "f5_cascade_diff": feature5_cascade_diff,
        "f6_same_line_diff": feature6_same_line_diff,
    }
    return feature_map
    
def evaluate_new (
    board: dict[Coord, CellState],
    color: PlayerColor,
    total_turn_count: int
) -> float:
    """
    Higher score means a better board for this player.
    """
    player_stacks: list[tuple[Coord, CellState]] = []
    opponent_stacks: list[tuple[Coord, CellState]] = []

    for coord, cell_state in board.items():
        if cell_state.color == color:
            player_stacks.append((coord, cell_state))
        else:
            opponent_stacks.append((coord, cell_state))

    # Quick win/lose checks.
    if not opponent_stacks:
        return 1000000.0
    if not player_stacks:
        return -1000000.0

    # Default weights.
    f1_weight = 0.5
    f2_weight = 0.5
    f3_weight = 0.8
    f4_weight = 1.0
    f5_weight = 1.0
    f6_weight = 0.2

    # Board-pattern adjustment (still needs cleanup later).
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

    # Compute each feature score.
    feature1_stack_num_diff = get_f1_score(opponent_stacks, player_stacks)
    feature2_stack_height_diff = get_f2_score(opponent_stacks,player_stacks)
    feature3_legal_action_diff = get_f3_score(board, color, total_turn_count)
    feature4_eat_diff = get_f4_score(opponent_stacks, player_stacks)
    feature5_cascade_diff = get_f5_score(board, opponent_stacks, player_stacks)
    feature6_same_line_diff = get_f6_score(opponent_stacks, player_stacks)

    return (
        + f1_weight * feature1_stack_num_diff
        + f2_weight * feature2_stack_height_diff
        + f3_weight * feature3_legal_action_diff
        + f4_weight * feature4_eat_diff
        + f5_weight * feature5_cascade_diff
        + f6_weight * feature6_same_line_diff
    )
