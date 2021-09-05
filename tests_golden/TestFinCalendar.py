###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from FinTestCases import FinTestCases, globalTestCaseMode
from financepy.utils.calendar import calendar, calendar_types
from financepy.utils.date import set_date_format, date_format_types
from financepy.utils.date import Date
import sys
sys.path.append("..")


testCases = FinTestCases(__file__, globalTestCaseMode)

###############################################################################


def test_Calendar():

    set_date_format(date_format_types.US_LONGEST)
    end_date = Date(31, 12, 2030)

    for calendar_type in calendar_types:

        testCases.banner("================================")
        testCases.banner("================================")

        testCases.header("CALENDAR", "HOLIDAY")
        testCases.print("STARTING", calendar_type)

        cal = calendar(calendar_type)
        next_date = Date(31, 12, 2020)

        while next_date < end_date:
            next_date = next_date.add_days(1)

            if next_date._d == 1 and next_date._m == 1:
                testCases.banner("================================")
#                print("=========================")

            is_holidayDay = cal.is_holiday(next_date)
            if is_holidayDay is True:
                testCases.print(cal, next_date)
#                print(cal, next_date)

    set_date_format(date_format_types.US_LONG)

###############################################################################


test_Calendar()
testCases.compareTestCases()
