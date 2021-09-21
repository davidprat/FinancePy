"""Copyright (C) Dominic O'Kane.

OLD code

@njit([float64(float64)], fastmath=True, cache=True)
 def normcdf(x: float):
    This is the Normal CDF function which forks to one of three of the
     implemented approximations. This is based on the choice of the fast flag
     variable. A value of 1 is the fast routine, 2 is the slow and 3 is the
     even slower integration scheme.

     return normcdf_fast(x)

      if fastFlag == 1:
          return normcdf_fast(x)
      elif fastFlag == 2:
          return normcdf_slow(x)
      elif fastFlag == 3:
          return normcdf_integrate(x)
      else:
               return 0.0


@vectorize([float64(float64)], fastmath=True, cache=True)
def N(x: float):
     This is the shortcut to the default Normal CDF function and currently
     is hardcoded to the fastest of the implemented routines. This is the most
     widely used way to access the Normal CDF.
     return normcdf_fast(x)

TODO: Move this somewhere else.
"""

# pylint: disable = E0401, R0913, RO914, C0200

from math import exp, fabs, log, sqrt
from typing import List

from numba import boolean, float64, int64, njit, vectorize
import numpy as np

from .error import finpy_error

PI = 3.14159265358979323846
INVROOT2PI = 0.3989422804014327

ONE_MILLION = 1_000_000
TEN_MILLION = 10_000_000
ONE_BILLION = 1_000_000_000


@njit(fastmath=True, cache=True)
def accrued_interpolator(tset: float,  # Settlement time in years
                         coupon_times: np.ndarray,
                         coupon_amounts: np.ndarray):
    """Fast calulation of accrued interest using an Actual/Actual type of convention.

    This does not calculate according to other conventions.

    TODO: NEED TO REVISIT THIS TODO
    """
    num_coupons = len(coupon_times)

    for i in range(1, num_coupons):

        pct = coupon_times[i - 1]
        nct = coupon_times[i]
        denom = (nct - pct)

        if pct <= tset < nct:
            accd_frac = (tset - pct) / denom
            accd_cpn = accd_frac * coupon_amounts[i]
            return accd_cpn
    # Very strange print after return
    return 0.0
    # print("t", tset)
    # print("CPN TIMES", coupon_times)
    # print("CPN AMNTS", coupon_amounts)
    # raise finpy_error("Failed to calculate accrued")


@njit(boolean(int64), fastmath=True, cache=True)
def is_leap_year(y: int) -> boolean:
    """Test whether year y is a leap year - if so return True, else False."""
    res = ((y % 4 == 0) and (y % 100 != 0) or (y % 400 == 0))
    return res


@njit(float64[:](float64[:], float64), fastmath=True, cache=True)
def scale(x: np.ndarray, factor: float):
    """Scale all of the elements of an array by the same amount factor."""
    x_scale = np.empty(len(x))
    for i in range(0, len(x)):
        x_scale[i] = x[i] * factor
    return x_scale


@njit(boolean(float64[:]), fastmath=True, cache=True)
def arr_is_monotonic(x: np.ndarray):
    """Check that an array of doubles is monotonic and strictly increasing."""
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            return False
    return True


@njit(fastmath=True, cache=True)
def arr_is_in_range(x: np.ndarray, lower: float, upper: float):
    """Check that all of the values of an array fall between a lower and upper bound."""
    for i in range(0, len(x)):
        if x[i] < lower:
            raise finpy_error("Value below lower.")
        if x[i] > upper:
            raise finpy_error("Value above upper.")


@njit(fastmath=True, cache=True)
def maximum(first_arr: np.ndarray, second_arr: np.ndarray):
    """Returns an array of the maximum of both arrays.

    TODO : Checking the length of the arrays are similar
    """
    arr_length = len(first_arr)
    out = [0.0] * arr_length

    for i in range(0, arr_length):
        if first_arr[i] > second_arr[i]:
            out[i] = first_arr[i]
        else:
            out[i] = second_arr[i]
    return out


@njit(float64[:](float64[:, :]), fastmath=True, cache=True)
def maxaxis(s: np.ndarray):
    """Perform a search for the vector of maximum values over an axis of a 2D Numpy Array."""
    shp = s.shape

    max_vector = np.empty(shp[0])

    for i in range(0, shp[0]):
        xmax = s[i, 0]
        for j in range(1, shp[1]):
            val = s[i, j]
            if val > xmax:
                xmax = val

        max_vector[i] = xmax

    return max_vector


@njit(float64[:](float64[:, :]), fastmath=True, cache=True)
def minaxis(to_search: np.ndarray):
    """Perform a search for the vector of minimum values over an axis of a 2D Numpy Array."""
    shp = to_search.shape

    min_vector = np.empty(shp[0])

    for i in range(0, shp[0]):
        xmin = to_search[i, 0]
        for j in range(1, shp[1]):
            x = to_search[i, j]
            if x < xmin:
                xmin = x

        min_vector[i] = xmin

    return min_vector


@njit(fastmath=True, cache=True)
def covar(first_array: np.ndarray, second_array: np.ndarray):
    """Calculate the Covariance of two arrays of numbers.

    TODO:
        -check that this works well for Numpy Arrays and add NUMBA function signature to code.
        -Do test of timings against Numpy.
    """
    array_length = len(first_array)

    sum_first_array = 0.0
    sum_second_array = 0.0

    sum_of_arrays = 0.0

    sum_first_array_squared = 0.0
    sum_second_array_quared = 0.0

    for i in range(0, array_length):
        sum_first_array = sum_first_array + first_array[i]
        sum_second_array = sum_second_array + second_array[i]
        sum_first_array_squared = sum_first_array_squared + first_array[i] ** 2
        sum_second_array_quared = sum_second_array_quared + second_array[i] ** 2
        sum_of_arrays = sum_of_arrays + first_array[i] * second_array[i]

    sum_first_array /= array_length
    sum_second_array /= array_length
    sum_first_array_squared /= array_length
    sum_second_array_quared /= array_length
    sum_of_arrays /= array_length

    first_array_variance = sum_first_array_squared - sum_first_array * sum_first_array
    second_array_variance = sum_second_array_quared - sum_second_array * sum_second_array
    covariance = sum_of_arrays - sum_first_array * sum_second_array

    matrix = [[0.0, 0.0], [0.0, 0.0]]
    matrix[0][0] = first_array_variance
    matrix[1][0] = covariance
    matrix[0][1] = covariance
    matrix[1][1] = second_array_variance
    return [[0.0, 0.0], [0.0, 0.0]]
    # [[first_array_variance, covar], [covar, second_array_variance]]


@njit(float64(float64, float64), fastmath=True, cache=True)
def pair_gcd(value_1: float, value_2: float) -> float:
    """Determine the Greatest Common Divisor of two integers using Euclid's algorithm.

    TODO:
        - compare this with math.gcd(a,b) for speed.
        - Also examine to see if I should not be declaring inputs as integers for NUMBA.
    """
    if value_1 == 0 or value_2 == 0:
        return 0

    try:
        ...
    except ZeroDivisionError:
        ...

    while value_2 != 0:
        temp = value_2
        factor = value_1 / value_2
        value_2 = value_1 - factor * value_2
        value_1 = temp

    return abs(value_1)


@njit(fastmath=True, cache=True)
def nprime(x: float) -> float:
    """Calculate the first derivative of the Cumulative Normal CDF.

    It is simply the PDF of the Normal Distribution.
    """
    inv_root_to_pi = 0.3989422804014327
    return np.exp(-x * x / 2.0) * inv_root_to_pi


@njit(fastmath=True, cache=True)
def heaviside(x: float) -> float:
    """Calculate the Heaviside function for x."""
    if x >= 0.0:
        return 1.0
    return 0.0


@njit(fastmath=True, cache=True)
def frange(start: int, stop: int, step: int):
    """ Calculate a range of values from start in steps of size step.

    Ends as soon as the value equals or exceeds stop.
    """
    res = []
    while start <= stop:
        res.append(start)
        start += step
    return res
    # return list(range(start=start, stop=stop, step=step))


@njit(fastmath=True, cache=True)
def normpdf(x: float):
    """Calculate the probability density function for a Gaussian (Normal) function at value x."""
    inv_root_to_pi = 0.3989422804014327
    return np.exp(-x * x / 2.0) * inv_root_to_pi


@njit(float64(float64), fastmath=True, cache=True)
def normal_cdf(val: float) -> float:
    """Fast Normal CDF function based on Hull OFAODS.

    4th Edition Page 252.
    This function is accurate to 6 decimal places.
    """
    a_1 = 0.319381530
    a_2 = -0.356563782
    a_3 = 1.781477937
    a_4 = -1.821255978
    a_5 = 1.330274429
    g = 0.2316419

    k = 1.0 / (1.0 + g * fabs(val))
    k_2 = k * k
    k_3 = k_2 * k
    k_4 = k_3 * k
    k_5 = k_4 * k

    if val >= 0.0:
        c = (a_1 * k + a_2 * k_2 + a_3 * k_3 + a_4 * k_4 + a_5 * k_5)
        phi = 1.0 - c * exp(-val * val / 2.0) * INVROOT2PI
    else:
        phi = 1.0 - normal_cdf(-val)

    return phi


@vectorize([float64(float64)], fastmath=True, cache=True)
def n_vect(x):
    """Retrieves the normal cdf."""
    return normal_cdf(x)


@vectorize([float64(float64)], fastmath=True, cache=True)
def n_prime_vect(arg):
    """Retrieves the n prime."""
    return nprime(arg)


@njit(float64(float64), fastmath=True, cache=True)
def normcdf_integrate(val: float):
    """Calculation of Normal Distribution CDF by simple integration.

    It can become exact in the limit of the number of steps tending
    towards infinity. This function is used for checking as it is slow
    since the number of integration steps is currently hardcoded to 10,000.
    """
    lower = -6.0
    upper = val
    num_steps = 10_000
    step = (upper - lower) / num_steps
    inv_root_to_pi = 0.3989422804014327

    val = lower
    func = exp(-val * val / 2.0)
    integral = func / 2.0

    for _ in range(0, num_steps - 1):
        val = val + step
        func = exp(-val * val / 2.0)
        integral += func

    val = val + step
    func = exp(-val * val / 2.0)
    integral += func / 2.0
    integral *= inv_root_to_pi * step
    return integral


@njit(float64(float64), fastmath=True, cache=True)
def normcdf_slow(val: float):
    """Calculation of Normal Distribution CDF accurate to 1d-15.

    This method is faster than integration but slower than other approximations.
    Reference: J.L. Schonfelder, Math Comp 32(1978), pp 1232-1240.
    """
    array = [0.0] * 25
    bips = 0.0

    rtwo = 1.4142135623731

    array[0] = 0.6101430819232
    array[1] = -0.434841272712578
    array[2] = 0.176351193643605
    array[3] = -6.07107956092494E-02
    array[4] = 1.77120689956941E-02
    array[5] = -4.32111938556729E-03
    array[6] = 8.54216676887099E-04
    array[7] = -1.27155090609163E-04
    array[8] = 1.12481672436712E-05
    array[9] = 3.13063885421821E-07
    array[10] = -2.70988068537762E-07
    array[11] = 3.07376227014077E-08
    array[12] = 2.51562038481762E-09
    array[13] = -1.02892992132032E-09
    array[14] = 2.99440521199499E-11
    array[15] = 2.60517896872669E-11
    array[16] = -2.63483992417197E-12
    array[17] = -6.43404509890636E-13
    array[18] = 1.12457401801663E-13
    array[19] = 1.72815333899861E-14
    array[20] = -4.26410169494238E-15
    array[21] = -5.45371977880191E-16
    array[22] = 1.58697607761671E-16
    array[23] = 2.0899837844334E-17
    array[24] = -5.900526869409E-18

    x_a = abs(val) / rtwo

    if x_a > 100:
        p = 0
    else:
        t = (8 * x_a - 30) / (4 * x_a + 15)
        bm = 0.0
        b = 0.0

        for i in range(0, 25):
            bips = b
            b = bm
            bm = t * b - bips + array[24 - i]

        p = exp(-x_a * x_a) * (bm - bips) / 4

    if val > 0:
        p = 1.0 - p

    return p


@njit(fastmath=True, cache=True)
def phi3(b_1: float, b_2: float, b_3: float, r_1_2: float, r_1_3: float, r_2_3: float):
    """Bivariate Normal CDF function to upper limits $b1$ and $b2$ which performs the innermost integral.

    This may need further refinement to ensure it is optimal.
    The current range of integration is from -7 and the integration steps are dx = 0.001.
    This may be excessive.
    """
    dx = 0.001
    lower_limit = -7
    upper_limit = b_1
    num_points = int((b_1 - lower_limit) / dx)
    dx = (upper_limit - lower_limit) / num_points
    x = lower_limit

    r12p = sqrt(1.0 - r_1_2 * r_1_2)
    r13p = sqrt(1.0 - r_1_3 * r_1_3)
    r123 = (r_2_3 - r_1_2 * r_1_3) / r12p / r13p

    v = 0.0

    for _ in range(1, num_points + 1):
        dp = normal_cdf(x + dx) - normal_cdf(x)
        h = (b_2 - r_1_2 * x) / r12p
        k = (b_3 - r_1_3 * x) / r13p
        bivariate = consistent_with_haug(h, k, r123)
        v = v + bivariate * dp
        x += dx

    return v


@njit(fastmath=True, cache=True)
def norminvcdf(p):
    """This algorithm computes the inverse Normal CDF and is based on the
    algorithm found at (http:#home.online.no/~pjacklam/notes/invnorm/)
    which is by John Herrero (3-Jan-03)."""

    # Define coefficients in rational approximations
    a_1 = -39.6968302866538
    a_2 = 220.946098424521
    a_3 = -275.928510446969
    a_4 = 138.357751867269
    a_5 = -30.6647980661472
    a_6 = 2.50662827745924

    b_1 = -54.4760987982241
    b_2 = 161.585836858041
    b_3 = -155.698979859887
    b_4 = 66.8013118877197
    b_5 = -13.2806815528857

    c_1 = -7.78489400243029E-03
    c_2 = -0.322396458041136
    c_3 = -2.40075827716184
    c_4 = -2.54973253934373
    c_5 = 4.37466414146497
    c_6 = 2.93816398269878

    d_1 = 7.78469570904146E-03
    d_2 = 0.32246712907004
    d_3 = 2.445134137143
    d_4 = 3.75440866190742

    inverse_cdf = 0.0

    # Define break-points
    p_low = 0.02425
    p_high = 1.0 - p_low

    # If argument out of bounds, raise error
    if p < 0.0 or p > 1.0:
        raise finpy_error("p must be between 0.0 and 1.0")

    if p == 0.0:
        p = 1e-10

    if p == 1.0:
        p = 1.0 - 1e-10

    if p < p_low:
        # Rational approximation for lower region
        q = sqrt(-2.0 * log(p))
        inverse_cdf = (((((c_1 * q + c_2) * q + c_3) * q + c_4) * q + c_5)
                       * q + c_6) / ((((d_1 * q + d_2) * q + d_3) * q + d_4) * q + 1.0)
    elif p <= p_high:
        # Rational approximation for lower region
        q = p - 0.5
        r = q * q
        inverse_cdf = (((((a_1 * r + a_2) * r + a_3) * r + a_4) * r + a_5) * r + a_6) * \
                      q / (((((b_1 * r + b_2) * r + b_3) * r + b_4) * r + b_5) * r + 1.0)
    elif p < 1.0:
        # Rational approximation for upper region
        q = sqrt(-2.0 * log(1 - p))
        inverse_cdf = -(((((c_1 * q + c_2) * q + c_3) * q + c_4) * q + c_5)
                        * q + c_6) / ((((d_1 * q + d_2) * q + d_3) * q + d_4) * q + 1.0)

    return inverse_cdf


@njit(fastmath=True, cache=True)
def consistent_with_haug(a, b, c):
    """This is used for consistency with Haug and its conciseness.

    Consider renaming phi2 to M."""
    return phi_2(a, b, c)


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def phi_2(h_1, h_k, val):
    """Drezner and Wesolowsky implementation of bi-variate normal."""

    #    if abs(r) > 0.9999999:
    #        raise FinError("Phi2: |Correlation| > 1")

    x_arr: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    w_arr: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    x_arr[0] = 0.04691008
    x_arr[1] = 0.23076534
    x_arr[2] = 0.5
    x_arr[3] = 0.76923466
    x_arr[4] = 0.95308992

    w_arr[0] = 0.018854042
    w_arr[1] = 0.038088059
    w_arr[2] = 0.0452707394
    w_arr[3] = 0.038088059
    w_arr[4] = 0.018854042

    h_2 = h_k
    h_12 = (h_1 * h_1 + h_2 * h_2) * 0.5
    b_v = 0.0

    if fabs(val) < 0.7 or fabs(h_1) > 35 or fabs(h_2) > 35:

        h_3 = h_1 * h_2

        for i in range(0, 5):
            r_1 = val * x_arr[i]
            rr2 = 1.0 - r_1 * r_1
            b_v = b_v + w_arr[i] * exp((r_1 * h_3 - h_12) / rr2) / sqrt(rr2)

        b_v = normal_cdf(h_1) * normal_cdf(h_2) + val * b_v
    else:
        r_2 = 1.0 - val * val
        r3 = sqrt(r_2)

        if val < 0.0:
            h_2 = -h_2

        h_3 = h_1 * h_2
        h_7 = exp(-h_3 * 0.5)

        if r_2 != 0.0:
            h_6 = abs(h_1 - h_2)
            h_5 = h_6 * h_6 * 0.5
            h_6 = h_6 / r3
            a_a = 0.5 - h_3 * 0.125
            a_b = 3.0 - 2.0 * a_a * h_5
            b_v = 0.13298076 * h_6 * a_b * \
                  normal_cdf(-h_6) - exp(-h_5 / r_2) * (a_b + a_a * r_2) * 0.053051647

            for i in range(0, 5):
                r_1 = r3 * x_arr[i]
                r_r = r_1 * r_1
                r_2 = sqrt(1.0 - r_r)
                b_v = b_v - w_arr[i] * exp(-h_5 / r_r) * \
                      (exp(-h_3 / (1.0 + r_2)) / r_2 / h_7 - 1.0 - a_a * r_r)

        if val > 0.0:
            b_v = b_v * r3 * h_7 + normal_cdf(min(h_1, h_2))
        else:
            if h_1 < h_2:
                b_v = -b_v * r3 * h_7
            else:
                b_v = -b_v * r3 * h_7 + normal_cdf(h_1) + normal_cdf(h_k) - 1.0

    return b_v


@njit(float64[:, :](float64[:, :]), cache=True, fastmath=True)
def cholesky(rho):
    """Numba-compliant wrapper around Numpy cholesky function."""
    chol = np.linalg.cholesky(rho)
    return chol


@njit(fastmath=True, cache=True)
def corr_matrix_generator(rho, dimension):
    """Utility function to generate a full rank n x n correlation matrix.

    With a flat correlation structure and value rho."""

    corr_matrix = np.zeros(shape=(dimension, dimension))
    for i in range(0, dimension):
        corr_matrix[i, i] = 1.0
        for j in range(0, i):
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    return corr_matrix
