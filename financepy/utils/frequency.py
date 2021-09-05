"""Copyright (C) Dominic O'Kane."""

from ..utils.error import finpy_error

###############################################################################

from enum import Enum


class frequency_types(Enum):
    """frequency_types simple enum."""
    SIMPLE = 0
    ANNUAL = 1
    SEMI_ANNUAL = 2
    TRI_ANNUAL = 3
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 99


def annual_frequency(freq_type: frequency_types):
    """ This is a function that takes in a Frequency Type and returns an
    float value for the number of times a year a payment occurs."""
    if isinstance(freq_type, frequency_types) is False:
        print("FinFrequency:", freq_type)
        raise finpy_error("Unknown frequency type")

    if freq_type == frequency_types.CONTINUOUS:
        return -1
    elif freq_type == frequency_types.SIMPLE:
        return 0.0
    elif freq_type == frequency_types.ANNUAL:
        return 1.0
    elif freq_type == frequency_types.SEMI_ANNUAL:
        return 2.0
    elif freq_type == frequency_types.TRI_ANNUAL:
        return 3.0
    elif freq_type == frequency_types.QUARTERLY:
        return 4.0
    elif freq_type == frequency_types.MONTHLY:
        return 12.0