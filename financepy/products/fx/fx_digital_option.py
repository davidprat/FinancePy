##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

from math import exp, log, sqrt
import numpy as np


from ...utils.math import normal_cdf
from ...utils.global_vars import g_days_in_year
from ...utils.error import finpy_error
# from ...products.equity.EquityOption import FinOption
from ...utils.date import Date
# from ...products.fx.FinFXModelTypes import FinFXModel
from ...models.black_scholes import BlackScholes
from ...utils.helpers import check_argument_types
from ...utils.global_types import option_types

###############################################################################


class FXDigitalOption:

    def __init__(self,
                 expiry_date: Date,
                 strike_price: float,  # 1 unit of foreign in domestic
                 currency_pair: str,  # FORDOM
                 option_type: option_types,
                 notional: float,
                 prem_currency: str):
        """ Create the FX Digital Option object. Inputs include expiry date,
        strike, currency pair, option type (call or put), notional and the
        currency of the notional. And adjustment for spot days is enabled. All
        currency rates must be entered in the price in domestic currency of
        one unit of foreign. And the currency pair should be in the form FORDOM
        where FOR is the foreign currency pair currency code and DOM is the
        same for the domestic currency. """

        check_argument_types(self.__init__, locals())

        self._expiry_date = expiry_date
        self._strike_price = float(strike_price)
        self._currency_pair = currency_pair
        self._option_type = option_type

        self._currency_pair = currency_pair
        self._forName = self._currency_pair[0:3]
        self._domName = self._currency_pair[3:6]

        if prem_currency != self._domName and prem_currency != self._forName:
            raise finpy_error("Notional currency not in currency pair.")

###############################################################################

    def value(self,
              valuation_date,
              spot_fx_rate,  # 1 unit of foreign in domestic
              dom_discount_curve,
              for_discount_curve,
              model):
        """ Valuation of a digital option using Black-Scholes model. This
        allows for 4 cases - first upper barriers that when crossed pay out
        cash (calls) and lower barriers than when crossed from above cause a
        cash payout (puts) PLUS the fact that the cash payment can be in
        domestic or foreign currency. """

        if type(valuation_date) == Date:
            spot_date = valuation_date.add_weekdays(self._spot_days)
            tdel = (self._delivery_date - spot_date) / g_days_in_year
            texp = (self._expiry_date - valuation_date) / g_days_in_year
        else:
            tdel = valuation_date
            texp = tdel

        if np.any(spot_fx_rate <= 0.0):
            raise finpy_error("spot_fx_rate must be greater than zero.")

        if np.any(tdel < 0.0):
            raise finpy_error("Option time to maturity is less than zero.")

        tdel = np.maximum(tdel, 1e-10)

        domDF = dom_discount_curve.df(tdel)
        rd = -np.log(domDF)/tdel

        forDF = for_discount_curve.df(tdel)
        rf = -np.log(forDF)/tdel

        S0 = spot_fx_rate
        K = self._strike_fx_rate

        if type(model) == BlackScholes:

            volatility = model._volatility
            lnS0k = log(S0 / K)
            den = volatility * sqrt(texp)
            v2 = volatility * volatility
            mu = rd - rf
            d2 = (lnS0k + (mu - v2 / 2.0) * tdel) / den

            if self._option_type == option_types.DIGITAL_CALL and \
                    self._forName == self._prem_currency:
                v = S0 * exp(-rf * tdel) * normal_cdf(d2)
            elif self._option_type == option_types.DIGITAL_PUT and \
                    self._forName == self._prem_currency:
                v = S0 * exp(-rf * tdel) * normal_cdf(-d2)
            if self._option_type == option_types.DIGITAL_CALL and \
                    self._domName == self._prem_currency:
                v = exp(-rd * tdel) * normal_cdf(d2)
            elif self._option_type == option_types.DIGITAL_PUT and \
                    self._domName == self._prem_currency:
                v = exp(-rd * tdel) * normal_cdf(-d2)
            else:
                raise finpy_error("Unknown option type")

            print("Prem_notional is an error ?")
            v = v * self.prem_notional

        return v

###############################################################################
