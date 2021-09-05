##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

##########################################################################
# THIS IS UNFINISHED
##########################################################################

from enum import Enum

from ...utils.day_count import day_count, day_count_types
from ...utils.frequency import frequency_types
from ...utils.calendar import calendar_types,  date_gen_rule_types
from ...utils.calendar import bus_day_adjust_types

##########################################################################


class IborConventions():

    def __init__(self,
                 currencyName: str,
                 indexName: str = "LIBOR"):

        if currencyName == "USD" and indexName == "LIBOR":
            self._spotLag = 2
            self._day_count_type = day_count_types.THIRTY_E_360_ISDA
            self._calendar_type = calendar_types.TARGET
        elif currencyName == "EUR" and indexName == "EURIBOR":
            self._spotLag = 2
            self._day_count_type = day_count_types.THIRTY_E_360_ISDA
            self._calendar_type = calendar_types.TARGET
        else:
            pass

###############################################################################
