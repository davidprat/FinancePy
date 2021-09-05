###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from financepy.utils.calendar import date_gen_rule_types
from financepy.utils.calendar import bus_day_adjust_types
from financepy.utils.day_count import day_count_types
from financepy.utils.calendar import calendar_types
from financepy.utils.frequency import frequency_types
from financepy.utils.date import Date, date_format_types
from financepy.products.bonds.bond_annuity import BondAnnuity

def test_SemiAnnual_BondAnnuity():

    settlement_date = Date(20, 6, 2018)
    face = 1000000

    #Semi-Annual Frequency
    maturity_date = Date(20, 6, 2019)
    coupon = 0.05
    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD
    basis_type = day_count_types.ACT_360
    

    annuity = BondAnnuity(maturity_date,
                          coupon,
                          freq_type,
                          calendar_type,
                          bus_day_adjust_type,
                          date_gen_rule_type,
                          basis_type,
                          face)

    annuity.calculate_payments(settlement_date) 

    assert len(annuity._flow_amounts) == 2 * 1 + 1
    assert len(annuity._flow_dates) == 2 * 1 + 1

    assert annuity._flow_dates[0] == settlement_date
    assert annuity._flow_dates[-1] == maturity_date

    assert annuity._flow_amounts[0] == 0.0
    assert round(annuity._flow_amounts[-1]) ==25278.0

    assert annuity.calc_accrued_interest(settlement_date) == 0.0

def test_Quarterly_BondAnnuity():

    settlement_date = Date(20, 6, 2018)
    face = 1000000

    #Quarterly Frequency
    maturity_date = Date(20, 6, 2028)
    coupon = 0.05
    freq_type = frequency_types.QUARTERLY
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD
    basis_type = day_count_types.ACT_360

    annuity = BondAnnuity(
        maturity_date,
        coupon,
        freq_type,
        calendar_type,
        bus_day_adjust_type,
        date_gen_rule_type,
        basis_type,
        face)

    annuity.calculate_payments(settlement_date) 

    assert len(annuity._flow_amounts) == 10 * 4 + 1
    assert len(annuity._flow_dates) == 10 * 4 + 1

    assert annuity._flow_dates[0] == settlement_date
    assert annuity._flow_dates[-1] == maturity_date

    assert annuity._flow_amounts[0] == 0.0
    assert round(annuity._flow_amounts[-1]) == 12778.0

    assert annuity.calc_accrued_interest(settlement_date) == 0.0

def test_Monthly_BondAnnuity():

    settlement_date = Date(20, 6, 2018)
    face = 1000000

    #Monthly Frequency
    maturity_date = Date(20, 6, 2028)
    coupon = 0.05
    freq_type = frequency_types.MONTHLY
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.BACKWARD
    basis_type = day_count_types.ACT_360

    annuity = BondAnnuity(maturity_date,
                          coupon,
                          freq_type,
                          calendar_type,
                          bus_day_adjust_type,
                          date_gen_rule_type,
                          basis_type,
                          face)

    annuity.calculate_payments(settlement_date)
    
    assert len(annuity._flow_amounts) == 10*12 + 1
    assert len(annuity._flow_dates) == 10*12 + 1

    assert annuity._flow_dates[0] == settlement_date
    assert annuity._flow_dates[-1] == maturity_date

    assert annuity._flow_amounts[0] == 0.0
    assert round(annuity._flow_amounts[-1]) == 4306.0

    assert annuity.calc_accrued_interest(settlement_date) == 0.0
    
def test_ForwardGen_BondAnnuity():

    settlement_date = Date(20, 6, 2018)
    face = 1000000

    maturity_date = Date(20, 6, 2028)
    coupon = 0.05
    freq_type = frequency_types.ANNUAL
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.FORWARD
    basis_type = day_count_types.ACT_360

    annuity = BondAnnuity(maturity_date,
                          coupon,
                          freq_type,
                          calendar_type,
                          bus_day_adjust_type,
                          date_gen_rule_type,
                          basis_type,
                          face)

    annuity.calculate_payments(settlement_date)

    assert len(annuity._flow_amounts) == 10 * 1 + 1
    assert len(annuity._flow_dates) == 10 * 1 + 1

    assert annuity._flow_dates[0] == settlement_date
    assert annuity._flow_dates[-1] == maturity_date

    assert round(annuity._flow_amounts[0]) == 0.0
    assert round(annuity._flow_amounts[-1]) == 50833.0

    assert annuity.calc_accrued_interest(settlement_date) == 0.0

def test_ForwardGenWithLongEndStub_BondAnnuity():

    settlement_date = Date(20, 6, 2018)
    face = 1000000

    maturity_date = Date(20, 6, 2028)
    coupon = 0.05
    freq_type = frequency_types.SEMI_ANNUAL
    calendar_type = calendar_types.WEEKEND
    bus_day_adjust_type = bus_day_adjust_types.FOLLOWING
    date_gen_rule_type = date_gen_rule_types.FORWARD
    basis_type = day_count_types.ACT_360

    annuity = BondAnnuity(maturity_date,
                          coupon, 
                          freq_type,
                          calendar_type,
                          bus_day_adjust_type,
                          date_gen_rule_type,
                          basis_type,
                          face)

    annuity.calculate_payments(settlement_date)

    assert len(annuity._flow_amounts) == 10 * 2 + 1
    assert len(annuity._flow_dates) == 10 * 2 + 1

    assert annuity._flow_dates[0] == settlement_date
    assert annuity._flow_dates[-1] == maturity_date

    assert round(annuity._flow_amounts[0]) == 0.0
    assert round(annuity._flow_amounts[-1]) == 25417.0

    assert annuity.calc_accrued_interest(settlement_date) == 0.0