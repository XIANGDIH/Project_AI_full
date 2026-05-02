from typing import TypeAlias
from referee.game import PlayerColor

SeenInfo: TypeAlias = tuple[int, PlayerColor]
SeenStates: TypeAlias = dict[tuple, SeenInfo]