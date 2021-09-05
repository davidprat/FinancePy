"""Copyright (C) Dominic O'Kane."""

from enum import Enum


class long_short(Enum):
    """Used to categorize Long or Short."""
    LONG = 1
    SHORT = 2


class option_types(Enum):
    """Used to categorize OptionTypes."""
    EUROPEAN_CALL = 1
    EUROPEAN_PUT = 2
    AMERICAN_CALL = 3
    AMERICAN_PUT = 4
    DIGITAL_CALL = 5
    DIGITAL_PUT = 6
    ASIAN_CALL = 7
    ASIAN_PUT = 8
    COMPOUND_CALL = 9
    COMPOUND_PUT = 10


class cap_floor(Enum):
    """Used to categorize Cap or Floors."""
    CAP = 1
    FLOOR = 2


class swap_types(Enum):
    """Used to categorize Swap types."""
    PAY = 1
    RECEIVE = 2


class exercise_types(Enum):
    """Used to categorize exercise types."""
    EUROPEAN = 1
    BERMUDAN = 2
    AMERICAN = 3


class solver_types(Enum):
    """Used to categorize exercise types."""
    CONJUGATE_GRADIENT = 0
    NELDER_MEAD = 1
    NELDER_MEAD_NUMBA = 2
