from enum import Enum
from typing import NamedTuple


class Direction(Enum):
    """
    Enumeration class with the possible movement directions
    """

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Point(NamedTuple):
    x: float
    y: float
