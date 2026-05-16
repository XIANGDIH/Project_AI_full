# This file is the optimization part

# This file contains the logic about the game play-strategy for the play phase.


from referee.game import PlayerColor, Coord, CellState, Action, EatAction, MoveAction, CascadeAction
from .evaluation_play import evaluate_new
from .rules import apply_action
from .types import SeenStates
from .helper import meaningful_cascade, copy_state, get_all_distance_to_opponent, get_all_distance_to_opponent_general
from .helper_play import BoardState, detect_board_state

# Basic helpers
def min_attackable_distance (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    if not opponent_stacks or not player_stacks:
        return float("inf")

    return min(get_all_distance_to_opponent(opponent_stacks, player_stacks), default=float("inf"))

def min_general_distance (opponent_stacks: list[tuple[Coord, CellState]], player_stacks: list[tuple[Coord, CellState]]) -> float:
    if not opponent_stacks or not player_stacks:
        return float("inf")

    return min(get_all_distance_to_opponent_general(opponent_stacks, player_stacks), default=float("inf"))

# Check whether the specific action is meaningless or not for action CASCADE and MOVE
def is_meaningless_cascade (board: dict[Coord, CellState], my_color: PlayerColor, action_to_be_applied: Action) -> bool:
    # If it's other actions
    if not isinstance(action_to_be_applied, CascadeAction):
        return False
    
    # Check whether the new cascade action is meaningful
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]

    attacker_coord = action_to_be_applied.coord
    attacker_state = CellState(my_color, board[attacker_coord].height)
    attacking_direction = action_to_be_applied.direction

    is_meaningful = meaningful_cascade(attacker_coord, attacker_state, opponent_stacks, attacking_direction)

    if is_meaningful:
        return False
    
    return True

def is_meaningless_movement_balanced (board: dict[Coord, CellState], my_color: PlayerColor, action_to_be_applied: Action) -> bool:
    # If it's other actions
    if not isinstance(action_to_be_applied, MoveAction):
        return False
    
    origin_opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]
    origin_player_stacks = [(c, s) for c, s in board.items() if s.color == my_color]
    origin_closest_general_dist = min_general_distance(origin_opponent_stacks, origin_player_stacks)

    next_board = copy_state(board)
    apply_action(next_board, my_color, action_to_be_applied)
    new_opponent_stacks = [(c, s) for c, s in next_board.items() if s.color == my_color.opponent]
    new_player_stacks = [(c, s) for c, s in next_board.items() if s.color == my_color]
    new_closest_general_dist = min_general_distance(new_opponent_stacks, new_player_stacks)

    if new_closest_general_dist > origin_closest_general_dist:
        return True

    return False

def filter_meaningful_actions (board: dict[Coord, CellState], color: PlayerColor, actions: list[Action], root_board_state: list[BoardState]) -> list[Action]:
    filtered_actions = []

    for action in actions:
        if is_meaningless_cascade(board, color, action):
            continue
        if BoardState.BALANCED in root_board_state and is_meaningless_movement_balanced(board, color, action):
            continue
        
        filtered_actions.append(action)

    return filtered_actions if filtered_actions else actions

# Order the legal actions
def order_actions (
    board: dict[Coord, CellState],
    my_color: PlayerColor,
    actions: list[Action],
    total_turn_count: int,
    seen_states: SeenStates | None = None,
    root_board_state: list[BoardState] | None = None
) -> list[Action]:
    # Filter the action that are meaningless
    actions = filter_meaningful_actions(board, my_color, actions, root_board_state)

    # The original board situation
    player_stacks = [(c, s) for c, s in board.items() if s.color == my_color]
    opponent_stacks = [(c, s) for c, s in board.items() if s.color == my_color.opponent]
    origin_player_num = len(player_stacks)
    origin_opponent_num = len(opponent_stacks)
    origin_meaningful_dist_min = min_attackable_distance(opponent_stacks, player_stacks)

    # Defensive check--should not happen
    if seen_states is None:
        seen_states = {}
    if root_board_state is None:
        root_board_state = detect_board_state(board, opponent_stacks, player_stacks)

    # Assign each action with a score, and order the actions according to the scores
    scored_actions = []

    for action in actions:
        # The new situation after applying this specific action
        next_board = copy_state(board)
        apply_action(next_board, my_color, action)
        new_player_stacks = [(c, s) for c, s in next_board.items() if s.color == my_color]
        new_opponent_stacks = [(c, s) for c, s in next_board.items() if s.color == my_color.opponent]
        new_player_num = len(new_player_stacks)
        new_opponent_num = len(new_opponent_stacks)


        # Baseline: Prefer actions that lead to better immediate evaluation
        evaluation_base = evaluate_new(
            next_board,
            my_color,
            total_turn_count + 1,
            seen_states,
            root_board_state
        )
        score = evaluation_base
        has_tactical_priority = False

        # Priority 1: Immediate opponent elimination
        # Prefer EAT actions
        # Consider this as the best for "saving efforts"
        if isinstance(action, EatAction):
            score += 1000
            has_tactical_priority = True
            if new_player_num < origin_player_num:
                score -= 500

        # Prefer successful (then meaningful) cascades
        if isinstance(action, CascadeAction):
            has_tactical_priority = True
            if new_opponent_num < origin_opponent_num:
                score += 500
                if new_player_num < origin_player_num:
                    score -= 250
            elif is_meaningless_cascade(board, my_color, action):
                score -= 20000
            # Priority 3:Meaningful cascade
            else:
                score += 100
            

        # Priority 2: Meaningful movement
        if isinstance(action, MoveAction) and not has_tactical_priority:
            new_meaningful_dist_min = min_attackable_distance(new_opponent_stacks, new_player_stacks)
            if new_meaningful_dist_min < origin_meaningful_dist_min:
                score += 100
            # Penalty: The action is meaningless
            else:
                score -= 50

        scored_actions.append((score, action))

    scored_actions.sort(reverse=True, key=lambda x: x[0])
    return [action for _, action in scored_actions]
