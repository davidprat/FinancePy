###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from FinTestCases import FinTestCases, globalTestCaseMode
from financepy.utils.calendar import date_gen_rule_types
from financepy.utils.calendar import bus_day_adjust_types
from financepy.utils.calendar import calendar_types
from financepy.utils.frequency import frequency_types
from financepy.utils.schedule import Schedule
from financepy.utils.date import Date
import sys
sys.path.append("..")


testCases = FinTestCases(__file__, globalTestCaseMode)

###############################################################################


def test_date_adjust():

    start_date = Date(28, 2, 2008)
    end_date = Date(28, 2, 2011)

    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.NONE
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD

    testCases.header("NO ADJUSTMENTS", "DATE")
    schedule = Schedule(start_date,
                        end_date,
                        freq_type,
                        calendar_type,
                        bus_day_adjust_type,
                        date_gen_rule_type)

    for dt in schedule._adjusted_dates:
        testCases.print("Date:", dt)

    testCases.banner("")
    testCases.header("NO WEEKENDS AND FOLLOWING", "DATE")
    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD

    schedule = Schedule(start_date,
                        end_date,
                        freq_type,
                        calendar_type,
                        bus_day_adjust_type,
                        date_gen_rule_type)

    for dt in schedule._adjusted_dates:
        testCases.print("Date:", dt)

    testCases.banner("")
    testCases.header("NO WEEKENDS AND MODIFIED FOLLOWING", "DATE")
    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.MODIFIED_FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD

    schedule = Schedule(start_date,
                        end_date,
                        freq_type,
                        calendar_type,
                        bus_day_adjust_type,
                        date_gen_rule_type)

    for dt in schedule._adjusted_dates:
        testCases.print("Date:", dt)

    testCases.banner("")
    testCases.header("NO WEEKENDS AND US HOLIDAYS AND MODIFIED FOLLOWING",
                     "DATE")
    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.UNITED_STATES
    bus_day_adjust_type = bus_day_adjust_types.MODIFIED_FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD

    start_date = Date(4, 7, 2008)
    end_date = Date(4, 7, 2011)

    schedule = Schedule(start_date,
                        end_date,
                        freq_type,
                        calendar_type,
                        bus_day_adjust_type,
                        date_gen_rule_type)

    for dt in schedule._adjusted_dates:
        testCases.print("Date:", dt)

###############################################################################


test_date_adjust()
testCases.compareTestCases()
