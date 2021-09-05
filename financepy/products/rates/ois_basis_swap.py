##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

from ...utils.error import finpy_error
from ...utils.date import Date
from ...utils.day_count import day_count_types
from ...utils.frequency import frequency_types
from ...utils.calendar import calendar_types, date_gen_rule_types
from ...utils.calendar import calendar, bus_day_adjust_types
from ...utils.helpers import check_argument_types, label_to_string
from ...utils.math import ONE_MILLION
from ...utils.global_types import swap_types
from ...market.curves.discount_curve import DiscountCurve

from .swap_float_leg import SwapFloatLeg

###############################################################################


class OISBasisSwap:
    """ Class for managing an Ibor-OIS basis swap contract. This is a
    contract in which a floating leg with one LIBOR tenor is exchanged for a 
    floating leg payment of an overnight index swap. There is no exchange of
    par. The contract is entered into at zero initial cost. The contract lasts
    from a start date to a specified maturity date.
    
    The value of the contract is the NPV of the two coupon streams. Discounting
    is done on a supplied discount curve which is separate from the discount from
    which the implied index rates are extracted. """
    
    def __init__(self,
                 effective_date: Date,  # Date interest starts to accrue
                 termination_date_or_tenor: (Date, str),  # Date contract ends
                 iborType: swap_types,
                 iborFreqType: frequency_types = frequency_types.QUARTERLY,
                 iborDayCountType: day_count_types  = day_count_types.THIRTY_E_360,
                 iborSpread: float = 0.0,
                 oisFreqType: frequency_types = frequency_types.QUARTERLY,
                 oisDayCountType: day_count_types = day_count_types.THIRTY_E_360,
                 oisSpread: float = 0.0,
                 oisPaymentLag: int = 0,
                 notional: float = ONE_MILLION,
                 calendar_type: calendar_types = calendar_types.WEEKEND,
                 bus_day_adjust_type: bus_day_adjust_types = bus_day_adjust_types.FOLLOWING,
                 date_gen_rule_type: date_gen_rule_types = date_gen_rule_types.BACKWARD):
        """ Create a Ibor basis swap contract giving the contract start
        date, its maturity, frequency and day counts on the two floating 
        legs and notional. The floating leg parameters have default
        values that can be overwritten if needed. The start date is contractual
        and is the same as the settlement date for a new swap. It is the date
        on which interest starts to accrue. The end of the contract is the
        termination date. This is not adjusted for business days. The adjusted
        termination date is called the maturity date. This is calculated. """

        check_argument_types(self.__init__, locals())

        if type(termination_date_or_tenor) == Date:
            self._termination_date = termination_date_or_tenor
        else:
            self._termination_date = effective_date.add_tenor(termination_date_or_tenor)

        calendar = calendar(calendar_type)
        self._maturity_date = calendar.adjust(self._termination_date,
                                             bus_day_adjust_type)

        if effective_date > self._maturity_date:
            raise finpy_error("Start date after maturity date")

        oisType = swap_types.PAY
        if iborType == swap_types.PAY:
            oisType = swap_types.RECEIVE
        
        principal = 0.0

        self._floatIborLeg = SwapFloatLeg(effective_date,
                                          self._termination_date,
                                          iborType,
                                          iborSpread,
                                          iborFreqType,
                                          iborDayCountType,
                                          notional,
                                          principal,
                                          0,
                                          calendar_type,
                                          bus_day_adjust_type,
                                          date_gen_rule_type)

        self._floatOISLeg = SwapFloatLeg(effective_date,
                                         self._termination_date,
                                         oisType,
                                         oisSpread,
                                         oisFreqType,
                                         oisDayCountType,
                                         notional,
                                         principal,
                                         oisPaymentLag,
                                         calendar_type,
                                         bus_day_adjust_type,
                                         date_gen_rule_type)

###############################################################################

    def value(self,
              valuation_date: Date,
              discount_curve: DiscountCurve,
              indexIborCurve: DiscountCurve = None,
              indexOISCurve: DiscountCurve = None,
              firstFixingRateLeg1=None,
              firstFixingRateLeg2=None):
        """ Value the interest rate swap on a value date given a single Ibor
        discount curve and an index curve for the Ibors on each swap leg. """

        if indexIborCurve is None:
            indexIborCurve = discount_curve

        if indexOISCurve is None:
            indexOISCurve = discount_curve

        floatIborLegValue = self._floatIborLeg.value(valuation_date,
                                                     discount_curve, 
                                                     indexIborCurve, 
                                                     firstFixingRateLeg1)

        floatOISLegValue = self._floatOISLeg.value(valuation_date,
                                                   discount_curve,
                                                   indexOISCurve,
                                                   firstFixingRateLeg2)

        value = floatIborLegValue + floatOISLegValue
        return value

###############################################################################

    def print_flows(self):
        """ Prints the fixed leg amounts without any valuation details. Shows
        the dates and sizes of the promised fixed leg flows. """

        self._floatIborLeg.print_payments()
        self._floatOISLeg.print_payments()

##########################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += self._floatIborLeg.__repr__()
        s += "\n"
        s += self._floatOISLeg.__repr__()
        return s

###############################################################################

    def _print(self):
        """ Print a list of the unadjusted coupon payment dates used in
        analytic calculations for the bond. """
        print(self)

###############################################################################
