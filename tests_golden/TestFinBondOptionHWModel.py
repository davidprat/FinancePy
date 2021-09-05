###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from FinTestCases import FinTestCases, globalTestCaseMode
from financepy.models.hw_tree import HWTree, FinHWEuropeanCalcType
from financepy.utils.global_types import option_types
from financepy.products.bonds.bond_option import BondOption
from financepy.utils.day_count import day_count_types
from financepy.utils.frequency import frequency_types
from financepy.products.bonds.bond import Bond
from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
from financepy.market.curves.discount_curve import DiscountCurve
from financepy.utils.date import Date
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
sys.path.append("..")


testCases = FinTestCases(__file__, globalTestCaseMode)

plotGraphs = False

###############################################################################


def test_BondOption():

    settlement_date = Date(1, 12, 2019)
    issue_date = Date(1, 12, 2018)
    maturity_date = settlement_date.add_tenor("10Y")
    coupon = 0.05
    freq_type = frequency_types.SEMI_ANNUAL
    accrual_type = day_count_types.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)

    times = np.linspace(0, 10.0, 21)
    dfs = np.exp(-0.05*times)
    dates = settlement_date.add_years(times)
    discount_curve = DiscountCurve(settlement_date, dates, dfs)

    expiry_date = settlement_date.add_tenor("18m")
    strike_price = 105.0
    face = 100.0

    ###########################################################################

    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]

    option_type = option_types.EUROPEAN_CALL

    testCases.header("LABEL", "VALUE")

    price = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)
    testCases.print("Fixed Income Price:", price)

    num_time_steps = 100

    testCases.banner("HW EUROPEAN CALL")
    testCases.header("STRIKE", "VALUE")

    for strike_price in strikes:

        sigma = 0.01
        a = 0.1

        bond_option = BondOption(bond, expiry_date, strike_price, face,
                                 option_type)

        model = HWTree(sigma, a, num_time_steps)
        v = bond_option.value(settlement_date, discount_curve, model)
        testCases.print(strike_price, v)

    ###########################################################################

    option_type = option_types.AMERICAN_CALL

    price = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)
    testCases.header("LABEL", "VALUE")
    testCases.print("Fixed Income Price:", price)

    testCases.banner("HW AMERICAN CALL")
    testCases.header("STRIKE", "VALUE")

    for strike_price in strikes:

        sigma = 0.01
        a = 0.1

        bond_option = BondOption(bond, expiry_date, strike_price, face,
                                 option_type)

        model = HWTree(sigma, a)
        v = bond_option.value(settlement_date, discount_curve, model)
        testCases.print(strike_price, v)

    ###########################################################################

    option_type = option_types.EUROPEAN_PUT
    testCases.banner("HW EUROPEAN PUT")
    testCases.header("STRIKE", "VALUE")

    price = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)

    for strike_price in strikes:

        sigma = 0.01
        a = 0.1

        bond_option = BondOption(bond, expiry_date, strike_price, face,
                                 option_type)

        model = HWTree(sigma, a)
        v = bond_option.value(settlement_date, discount_curve, model)
        testCases.print(strike_price, v)

    ###########################################################################

    option_type = option_types.AMERICAN_PUT
    testCases.banner("HW AMERICAN PUT")
    testCases.header("STRIKE", "VALUE")

    price = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)

    for strike_price in strikes:

        sigma = 0.02
        a = 0.1

        bond_option = BondOption(bond, expiry_date, strike_price, face,
                                 option_type)

        model = HWTree(sigma, a)
        v = bond_option.value(settlement_date, discount_curve, model)
        testCases.print(strike_price, v)

###############################################################################


def test_BondOptionEuropeanConvergence():

    # CONVERGENCE TESTS
    # COMPARE AMERICAN TREE VERSUS JAMSHIDIAN IN EUROPEAN LIMIT TO CHECK THAT
    # TREE HAS BEEN CORRECTLY CONSTRUCTED. FIND VERY GOOD AGREEMENT.

    # Build discount curve
    settlement_date = Date(1, 12, 2019)
    discount_curve = DiscountCurveFlat(settlement_date, 0.05,
                                       frequency_types.CONTINUOUS)

    # Bond details
    issue_date = Date(1, 12, 2015)
    maturity_date = Date(1, 12, 2020)
    coupon = 0.05
    freq_type = frequency_types.ANNUAL
    accrual_type = day_count_types.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)

    # Option Details - put expiry in the middle of a coupon period
    expiry_date = Date(1, 3, 2020)
    strike_price = 100.0
    face = 100.0

    timeSteps = range(100, 400, 100)
    strike_price = 100.0

    testCases.header("TIME", "N", "PUT_JAM", "PUT_TREE",
                     "CALL_JAM", "CALL_TREE")

    for num_time_steps in timeSteps:

        sigma = 0.05
        a = 0.1

        start = time.time()
        option_type = option_types.EUROPEAN_PUT

        bond_option1 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)
        model1 = HWTree(sigma, a, num_time_steps)
        v1put = bond_option1.value(settlement_date, discount_curve, model1)

        bond_option2 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model2 = HWTree(sigma, a, num_time_steps,
                        FinHWEuropeanCalcType.EXPIRY_ONLY)
        v2put = bond_option2.value(settlement_date, discount_curve, model2)

        option_type = option_types.EUROPEAN_CALL

        bond_option1 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model1 = HWTree(sigma, a, num_time_steps)
        v1call = bond_option1.value(settlement_date, discount_curve, model1)

        bond_option2 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model2 = HWTree(sigma, a, num_time_steps,
                        FinHWEuropeanCalcType.EXPIRY_TREE)
        v2call = bond_option2.value(settlement_date, discount_curve, model2)

        end = time.time()
        period = end - start
        testCases.print(period, num_time_steps, v1put, v2put, v1call, v2call)

###############################################################################


def test_BondOptionAmericanConvergenceONE():

    # Build discount curve
    settlement_date = Date(1, 12, 2019)
    discount_curve = DiscountCurveFlat(settlement_date, 0.05)

    # Bond details
    issue_date = Date(1, 9, 2014)
    maturity_date = Date(1, 9, 2025)
    coupon = 0.05
    freq_type = frequency_types.SEMI_ANNUAL
    accrual_type = day_count_types.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)

    # Option Details
    expiry_date = Date(1, 12, 2020)
    strike_price = 100.0
    face = 100.0

    testCases.header("TIME", "N", "PUT_AMER", "PUT_EUR",
                     "CALL_AME", "CALL_EUR")

    timeSteps = range(100, 500, 100)

    for num_time_steps in timeSteps:

        sigma = 0.05
        a = 0.1

        start = time.time()

        option_type = option_types.AMERICAN_PUT
        bond_option1 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model1 = HWTree(sigma, a, num_time_steps)
        v1put = bond_option1.value(settlement_date, discount_curve, model1)

        option_type = option_types.EUROPEAN_PUT
        bond_option2 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model2 = HWTree(sigma, a, num_time_steps,
                        FinHWEuropeanCalcType.EXPIRY_ONLY)
        v2put = bond_option2.value(settlement_date, discount_curve, model2)

        option_type = option_types.AMERICAN_CALL
        bond_option1 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model1 = HWTree(sigma, a, num_time_steps)
        v1call = bond_option1.value(settlement_date, discount_curve, model1)

        option_type = option_types.EUROPEAN_CALL
        bond_option2 = BondOption(bond, expiry_date, strike_price, face,
                                  option_type)

        model2 = HWTree(sigma, a, num_time_steps,
                        FinHWEuropeanCalcType.EXPIRY_TREE)
        v2call = bond_option2.value(settlement_date, discount_curve, model2)

        end = time.time()

        period = end - start

        testCases.print(period, num_time_steps, v1put, v2put, v1call, v2call)

###############################################################################


def test_BondOptionAmericanConvergenceTWO():

    # Build discount curve
    settlement_date = Date(1, 12, 2019)
    discount_curve = DiscountCurveFlat(settlement_date, 0.05)

    # Bond details
    issue_date = Date(1, 12, 2015)
    maturity_date = settlement_date.add_tenor("10Y")
    coupon = 0.05
    freq_type = frequency_types.SEMI_ANNUAL
    accrual_type = day_count_types.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)
    expiry_date = settlement_date.add_tenor("18m")
    face = 100.0

    spotValue = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)
    testCases.header("LABEL", "VALUE")
    testCases.print("BOND PRICE", spotValue)

    testCases.header("TIME", "N", "EUR_CALL", "AMER_CALL",
                     "EUR_PUT", "AMER_PUT")

    sigma = 0.01
    a = 0.1
    hwModel = HWTree(sigma, a)
    K = 102.0

    vec_ec = []
    vec_ac = []
    vec_ep = []
    vec_ap = []

    num_stepsVector = range(100, 500, 100)

    for num_steps in num_stepsVector:
        hwModel = HWTree(sigma, a, num_steps)

        start = time.time()

        europeanCallBondOption = BondOption(bond, expiry_date, K, face,
                                            option_types.EUROPEAN_CALL)

        v_ec = europeanCallBondOption.value(settlement_date, discount_curve,
                                            hwModel)

        americanCallBondOption = BondOption(bond, expiry_date, K, face,
                                            option_types.AMERICAN_CALL)

        v_ac = americanCallBondOption.value(settlement_date, discount_curve,
                                            hwModel)

        europeanPutBondOption = BondOption(bond, expiry_date, K, face,
                                           option_types.EUROPEAN_PUT)

        v_ep = europeanPutBondOption.value(settlement_date, discount_curve,
                                           hwModel)

        americanPutBondOption = BondOption(bond, expiry_date, K, face,
                                           option_types.AMERICAN_PUT)

        v_ap = americanPutBondOption.value(settlement_date, discount_curve,
                                           hwModel)

        end = time.time()
        period = end - start

        testCases.print(period, num_steps, v_ec, v_ac, v_ep, v_ap)

        vec_ec.append(v_ec)
        vec_ac.append(v_ac)
        vec_ep.append(v_ep)
        vec_ap.append(v_ap)

    if plotGraphs:
        plt.figure()
        plt.plot(num_stepsVector, vec_ac, label="American Call")
        plt.legend()

        plt.figure()
        plt.plot(num_stepsVector, vec_ap, label="American Put")
        plt.legend()


###############################################################################

def test_BondOptionZEROVOLConvergence():

    # Build discount curve
    settlement_date = Date(1, 9, 2019)
    rate = 0.05
    discount_curve = DiscountCurveFlat(
        settlement_date, rate, frequency_types.ANNUAL)

    # Bond details
    issue_date = Date(1, 9, 2014)
    maturity_date = Date(1, 9, 2025)
    coupon = 0.06
    freq_type = frequency_types.ANNUAL
    accrual_type = day_count_types.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)

    # Option Details
    expiry_date = Date(1, 12, 2021)
    face = 100.0

    dfExpiry = discount_curve.df(expiry_date)
    fwdCleanValue = bond.clean_price_from_discount_curve(
        expiry_date, discount_curve)
#    fwdFullValue = bond.full_price_from_discount_curve(expiry_date, discount_curve)
#    print("BOND FwdCleanBondPx", fwdCleanValue)
#    print("BOND FwdFullBondPx", fwdFullValue)
#    print("BOND Accrued:", bond._accrued_interest)

    spotCleanValue = bond.clean_price_from_discount_curve(
        settlement_date, discount_curve)

    testCases.header("STRIKE", "STEPS",
                     "CALL_INT", "CALL_INT_PV", "CALL_EUR", "CALL_AMER",
                     "PUT_INT", "PUT_INT_PV", "PUT_EUR", "PUT_AMER")

    num_time_steps = range(100, 1000, 100)
    strike_prices = [90, 100, 110, 120]

    for strike_price in strike_prices:

        callIntrinsic = max(spotCleanValue - strike_price, 0)
        putIntrinsic = max(strike_price - spotCleanValue, 0)
        callIntrinsicPV = max(fwdCleanValue - strike_price, 0) * dfExpiry
        putIntrinsicPV = max(strike_price - fwdCleanValue, 0) * dfExpiry

        for num_steps in num_time_steps:

            sigma = 0.0000001
            a = 0.1
            model = HWTree(sigma, a, num_steps)

            option_type = option_types.EUROPEAN_CALL
            bond_option1 = BondOption(
                bond, expiry_date, strike_price, face, option_type)
            v1 = bond_option1.value(settlement_date, discount_curve, model)

            option_type = option_types.AMERICAN_CALL
            bond_option2 = BondOption(
                bond, expiry_date, strike_price, face, option_type)
            v2 = bond_option2.value(settlement_date, discount_curve, model)

            option_type = option_types.EUROPEAN_PUT
            bond_option3 = BondOption(
                bond, expiry_date, strike_price, face, option_type)
            v3 = bond_option3.value(settlement_date, discount_curve, model)

            option_type = option_types.AMERICAN_PUT
            bond_option4 = BondOption(
                bond, expiry_date, strike_price, face, option_type)
            v4 = bond_option4.value(settlement_date, discount_curve, model)

            testCases.print(strike_price, num_steps,
                            callIntrinsic, callIntrinsicPV, v1, v2,
                            putIntrinsic, putIntrinsicPV, v3, v4)

###############################################################################


test_BondOptionZEROVOLConvergence()
test_BondOption()
test_BondOptionEuropeanConvergence()
test_BondOptionAmericanConvergenceONE()
test_BondOptionAmericanConvergenceTWO()
testCases.compareTestCases()
