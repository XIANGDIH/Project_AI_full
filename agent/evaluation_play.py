# This file contains the logic about finding "EVAL" for the play phase.
# We reuse old heuristic ideas, but adapt them for two-player evaluation.


from referee.game import PlayerColor, Coord, Direction, CellState, BOARD_N

from .helper import get_same_direction, successful_cascade, is_adjacent
from .helper_play import BoardState, detect_board_state


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