from referee.game import PlayerColor, Coord, CellState
from agent.program import Agent


def make_test_board():
    board = {}

    board[Coord(3, 3)] = CellState(PlayerColor.RED, 3)
    board[Coord(4, 4)] = CellState(PlayerColor.RED, 2)

    board[Coord(3, 5)] = CellState(PlayerColor.BLUE, 2)
    board[Coord(5, 4)] = CellState(PlayerColor.BLUE, 3)

    return board


def main():
    actions = []

    for _ in range(100):
        agent = Agent(PlayerColor.RED)

        # Force the agent into play phase
        agent._turn_count = 4
        agent._total_turn_count = 10

        # Give it the same board every time
        agent._board = make_test_board()

        action = agent.action()
        actions.append(str(action))

    print(set(actions))


if __name__ == "__main__":
    main()