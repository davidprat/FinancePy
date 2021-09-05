###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from FinTestCases import FinTestCases, globalTestCaseMode
from financepy.utils.frequency import frequency_types
from financepy.utils.date import Date
from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
import sys
sys.path.append("..")


testCases = FinTestCases(__file__, globalTestCaseMode)

##############################################################################


def test_FinFlatCurve():

    curve_date = Date(1, 1, 2019)
    months = range(1, 60, 3)
    dates = curve_date.add_months(months)
    testCases.header("COMPOUNDING", "DFS")
    compounding = frequency_types.CONTINUOUS

    flat_curve = DiscountCurveFlat(curve_date, 0.05, compounding)
    dfs = flat_curve.df(dates)
    testCases.print(compounding, dfs)

    compounding = frequency_types.ANNUAL
    flat_curve = DiscountCurveFlat(curve_date, 0.05, compounding)
    dfs = flat_curve.df(dates)
    testCases.print(compounding, dfs)

    compounding = frequency_types.SEMI_ANNUAL
    flat_curve = DiscountCurveFlat(curve_date, 0.05, compounding)
    dfs = flat_curve.df(dates)
    testCases.print(compounding, dfs)

    compounding = frequency_types.QUARTERLY
    flat_curve = DiscountCurveFlat(curve_date, 0.05, compounding)
    dfs = flat_curve.df(dates)
    testCases.print(compounding, dfs)

    compounding = frequency_types.MONTHLY
    flat_curve = DiscountCurveFlat(curve_date, 0.05, compounding)
    dfs = flat_curve.df(dates)
    testCases.print(compounding, dfs)

###############################################################################


test_FinFlatCurve()
testCases.compareTestCases()
