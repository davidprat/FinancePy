"""Copyright (C) Dominic O'Kane.

A useful source for these definitions can be found at
https://developers.opengamma.com/quantitative-research/Interest-Rate-Instruments-and-Market-Conventions.pdf
and https://en.wikipedia.org/wiki/Day_count_convention
http://www.fairmat.com/documentation/usermanual/topics/download/mediawiki/index.php/Day_Count_Conventions.htm
http://www.eclipsesoftware.biz/DayCountConventions.html

    THIRTY_360_BOND = 1  # 30E/360 ISDA 2006 4.16f, German, Eurobond(ISDA 2000)
    THIRTY_E_360 = 2  # ISDA 2006 4.16(g) 30/360 ISMA, ICMA
    THIRTY_E_360_ISDA = 3  # ISDA 2006 4.16(h)
    THIRTY_E_PLUS_360 = 4  # ROLLS D2 TO NEXT MONTH IF D2 = 31
    ACT_ACT_ISDA = 5  # SPLITS ACCRUAL PERIOD INTO LEAP YEAR AND NON LEAP YEAR
    ACT_ACT_ICMA = 6  # METHOD FOR ALL US TREASURY NOTES AND BONDS
    ACT_365F = 7  # Denominator is always Fixed at 365, even in a leap year
    ACT_360 = 8
    ACT_365L = 9  # the 29 Feb is counted if it is in the date range

"""
from .date import Date, monthDaysLeapYear, monthDaysNotLeapYear, datediff
from .date import is_leap_year
from .error import finpy_error
from .frequency import frequency_types, annual_frequency
from .global_vars import g_days_in_year

from enum import Enum


def is_last_day_of_feb(dt: Date):
    # Return true if we are on the last day of February
    if dt._m == 2:
        is_leap: bool = is_leap_year(dt._y)
        if is_leap and dt._d == 29:
            return True
        if not is_leap and dt._d == 28:
            return True
    else:
        return False


class day_count_types(Enum):
    THIRTY_360_BOND = 1
    THIRTY_E_360 = 2
    THIRTY_E_360_ISDA = 3
    THIRTY_E_PLUS_360 = 4
    ACT_ACT_ISDA = 5
    ACT_ACT_ICMA = 6
    ACT_365F = 7
    ACT_360 = 8
    ACT_365L = 9
    SIMPLE = 10  # actual divided by gDaysInYear


class day_count:
    """ Calculate the fractional day count between two dates according to a
    specified day count convention. """

    def __init__(self,
                 dccType: day_count_types):
        """ Create Day Count convention by passing in the Day Count Type. """

        if dccType not in day_count_types:
            raise finpy_error("Need to pass FinDayCountType")

        self._type = dccType

    ###############################################################################

    def year_frac(self,
                  dt_1: Date,  # Start of coupon period
                  dt_2: Date,  # Settlement (for bonds) or period end(swaps)
                  dt_3: Date = None,  # End of coupon period for accrued
                  freq_type: frequency_types = frequency_types.ANNUAL,
                  is_termination_date: bool = False):  # Is dt2 a termination date
        """ This method performs two functions:

        1) It calculates the year fraction between dates dt1 and dt2 using the
        specified day count convention which is useful for calculating year
        fractions for Libor products whose flows are day count adjusted. In
        this case we will set dt3 to be None
        2) This function is also for calculating bond accrued where dt1 is the
        last coupon date, dt2 is the settlement date of the bond and date dt3
        must be set to the next coupon date. You will also need to provide a
        coupon frequency for some conventions.

        Note that if the date is intraday, i.e. hh,mm and ss do not equal zero
        then that is used in the calculation of the year frac. This avoids 
        discontinuities for short dated intra day products. It should not
        affect normal dates for which hh=mm=ss=0.
        
        This seems like a useful source:
        https://www.eclipsesoftware.biz/DayCountConventions.html
        Wikipedia also has a decent survey of the conventions
        https://en.wikipedia.org/wiki/Day_count_convention
        and
        http://data.cbonds.info/files/cbondscalc/Calculator.pdf
        """

        d_1 = dt_1._d
        m_1 = dt_1._m
        y_1 = dt_1._y

        d_2 = dt_2._d
        m_2 = dt_2._m
        y_2 = dt_2._y

        if self._type == day_count_types.THIRTY_360_BOND:
            # It is in accordance with section 4.16(f) of ISDA 2006 Definitions
            # Also known as 30/360, Bond Basis, 30A/360, 30-360 US Municipal
            # This method does not consider February as a special case.

            if d_1 == 31:
                d_1 = 30

            if d_2 == 31 and d_1 == 30:
                d_2 = 30

            num = 360 * (y_2 - y_1) + 30 * (m_2 - m_1) + (d_2 - d_1)
            den = 360
            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.THIRTY_E_360:
            # This is in section 4.16(g) of ISDA 2006 Definitions
            # Also known as 30/360 Eurobond, 30/360 ISMA, 30/360 ICMA,
            # 30/360 European, Special German, Eurobond basis (ISDA 2006)
            # Unlike 30/360 BOND the adjustment to dt2 does not depend on dt1

            if d_1 == 31:
                d_1 = 30

            if d_2 == 31:
                d_2 = 30

            num = 360 * (y_2 - y_1) + 30 * (m_2 - m_1) + (d_2 - d_1)
            den = 360
            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.THIRTY_E_360_ISDA:
            # This is 30E/360 (ISDA 2000), 30E/360 (ISDA) section 4.16(h)
            # of ISDA 2006 Definitions, German, Eurobond basis (ISDA 2000)

            if d_1 == 31:
                d_1 = 30

            lastDayOfFeb1 = is_last_day_of_feb(dt_1)
            if lastDayOfFeb1 is True:
                d_1 = 30

            if d_2 == 31:
                d_2 = 30

            lastDayOfFeb2 = is_last_day_of_feb(dt_2)
            if lastDayOfFeb2 is True and is_termination_date is False:
                d_2 = 30

            num = 360 * (y_2 - y_1) + 30 * (m_2 - m_1) + (d_2 - d_1)
            den = 360
            acc_factor = num / den
            return acc_factor, num, den

        elif self._type == day_count_types.THIRTY_E_PLUS_360:

            if d_1 == 31:
                d_1 = 30

            if d_2 == 31:
                m_2 = m_2 + 1  # May roll to 13 but we are doing a difference
                d_2 = 1

            num = 360 * (y_2 - y_1) + 30 * (m_2 - m_1) + (d_2 - d_1)
            den = 360
            acc_factor = num / den
            return acc_factor, num, den

        elif self._type == day_count_types.ACT_ACT_ISDA:

            if is_leap_year(y_1):
                denom_1 = 366
            else:
                denom_1 = 365

            if is_leap_year(y_2):
                denom_2 = 366
            else:
                denom_2 = 365

            if y_1 == y_2:
                num = dt_2 - dt_1
                den = denom_1
                acc_factor = (dt_2 - dt_1) / denom_1
                return (acc_factor, num, den)
            else:
                days_year_1 = datediff(dt_1, Date(1, 1, y_1 + 1))
                days_year_2 = datediff(Date(1, 1, y_2), dt_2)
                acc_factor_1 = days_year_1 / denom_1
                acc_factor_2 = days_year_2 / denom_2
                year_diff = y_2 - y_1 - 1.0
                # Note that num/den does not equal acc_factor
                # I do need to pass num back
                num = days_year_1 + days_year_2
                den = denom_1 + denom_2
                acc_factor = acc_factor_1 + acc_factor_2 + year_diff
                return (acc_factor, num, den)

        elif self._type == day_count_types.ACT_ACT_ICMA:

            freq = annual_frequency(freq_type)

            if dt_3 is None or freq is None:
                raise finpy_error("ACT_ACT_ICMA requires three dates and a freq")

            num = dt_2 - dt_1
            den = freq * (dt_3 - dt_1)
            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.ACT_365F:

            num = dt_2 - dt_1
            den = 365
            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.ACT_360:

            num = dt_2 - dt_1
            den = 360
            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.ACT_365L:

            # The ISDA calculator sheet appears to split this across the
            # non-leap and the leap year which I do not see in any conventions.

            frequency = annual_frequency(freq_type)

            if dt_3 is None:
                y3 = y_2
            else:
                y3 = dt_3._y

            num = dt_2 - dt_1
            den = 365

            if is_leap_year(y_1):
                feb29 = Date(29, 2, y_1)
            elif is_leap_year(y3):
                feb29 = Date(29, 2, y3)
            else:
                feb29 = Date(1, 1, 1900)

            if frequency == 1:
                if feb29 > dt_1 and feb29 <= dt_3:
                    den = 366
            else:
                if is_leap_year(y3) is True:
                    den = 366

            acc_factor = num / den
            return (acc_factor, num, den)

        elif self._type == day_count_types.SIMPLE:

            num = dt_2 - dt_1
            den = g_days_in_year
            acc_factor = num / den
            return (acc_factor, num, den)

        else:

            raise finpy_error(str(self._type) +
                           " is not one of DayCountTypes")

    def __repr__(self):
        """ Returns the calendar type as a string."""
        return str(self._type)
