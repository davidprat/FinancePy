"""Copyright (C) Saeed Amen,Dominic O'Kane.

from https://quanteconpy.readthedocs.io/en/latest/_modules/quantecon/optimize/root_finding.html #####################

"""

from collections import namedtuple
import operator

from numba import njit
import numpy as np

from .error import finpy_error

_econverged = 0
_econverr = -1

_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps

results = namedtuple('results', 'root function_calls iterations converged')


@njit(cache=True, fastmath=True)
def _results(r):
    """Select from a tuple of(root, funccalls, iterations, flag).

    results(x, funcalls, iterations, flag == 0)
    """
    return list(r)[0]


@njit(fastmath=True, cache=True)
def newton_secant(func, x_0, args=(), tol: float = 1.48e-8, maxiter: int = 50, disp: bool = True):
    """
    Find a zero from the secant method using the jitted version of
    Scipy's secant method.

    DO NOT TOUCH THIS FUNCTION AS IT IS USED IN FX VOL CALIBRATION !!!!!!!!!
    IT NEEDS TO PASS IN ARGS AS A TUPLE AS ONE OF THE ARGS IS AN NDARRAY

    Note that `func` must be jitted via Numba.

    Parameters
    ----------
    func : callable and jitted
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x_0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple
        A namedtuple containing the following items:
        ::

            root - Estimated location where function is zero.
            function_calls - Number of times the function was called.
            iterations - Number of iterations needed to find the root.
            converged - True if the routine converged
    """

    if tol <= 0.0:
        raise finpy_error("Tolerance should be positive.")

    if maxiter < 1:
        raise finpy_error("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    eps = 1e-4
    p0 = 1.0 * x_0
    funcalls = 0
    status = _econverr

    p1 = x_0 * (1.0 + eps)

    if p1 > 0.0:
        p1 = p1 + eps
    else:
        p1 = p1 - eps

    q0 = func(p0, *args)
    funcalls += 1
    q1 = func(p1, *args)
    funcalls += 1

    if np.abs(q1) < np.abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0

    for _ in range(maxiter):

        if q1 == q0:
            if p1 != p0:
                raise finpy_error("Tolerance reached")

            p = (p1 + p0) / 2.0
            status = _econverged
            break
        else:
            if np.abs(q1) > np.abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0)

        if np.abs(p - p1) < tol:
            status = _econverged
            return p

        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, *args)
        funcalls += 1

    if disp and status == _econverr:
        msg = "Failed to converge"
        raise finpy_error(msg)

    return p


# @jit
def newton(func, x_0, fprime=None, args=None, tol=1.48e-8, max_iter=50,
           fprime_2=None, x_1=None, rtol=0.0, full_output=False, disp=False):
    """TAKEN FROM SCIPY.
    
    Find a zero of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.
    Find a zero of the function `func` given a nearby starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used. If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.
    If `x0` is a sequence with more than one item, then `newton` returns an
    array, and `func` must be vectorized and return a sequence or array of the
    same shape as its first argument. If `fprime` or `fprime2` is given, then
    its return must also have the same shape.
    Parameters
    ----------
    func : callable
        The function whose zero is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x_0 : float, sequence, or ndarray
        An initial estimate of the zero that should be somewhere near the
        actual zero. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    max_iter : int, optional
        Maximum number of iterations.
    fprime_2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x_1 : float, optional
        Another estimate of the zero that should be somewhere near the
        actual zero. Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    full_output : bool, optional
        If `full_output` is False (default), the root is returned.
        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
        is the root and ``r`` is a `RootResults` object.
        If True and `x0` is non-scalar, the return value is ``(x, converged,
        zero_der)`` (see Returns section for details).
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, with
        the error message containing the number of iterations and current
        function value. Otherwise, the convergence status is recorded in a
        `RootResults` return object.
        Ignored if `x0` is not scalar.
        *Note: this has little to do with displaying, however,
        the `disp` keyword cannot be renamed for backwards compatibility.*
    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r : `RootResults`, optional
        Present if ``full_output=True`` and `x0` is scalar.
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.
    converged : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements converged successfully.
    zero_der : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements had a zero derivative.
    See Also
    --------
    brentq, brenth, ridder, bisect
    fsolve : find zeros in N dimensions.
    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic. This means that if the function is well-behaved
    the actual error in the estimated zero after the nth iteration
    is approximately the square (cube for Halley) of the error
    after the (n-1)th step. However, the stopping criterion used
    here is the step size and there is no guarantee that a zero
    has been found. Consequently, the result should be verified.
    Safer algorithms are brentq, brenth, ridder, and bisect,
    but they all require that the root first be bracketed in an
    interval where the function changes sign. The brentq algorithm
    is recommended for general use in one dimensional problems
    when such an interval has been found.
    When `newton` is used with arrays, it is best suited for the following
    types of problems:
    * The initial guesses, `x0`, are all relatively the same distance from
      the roots.
    * Some or all of the extra arguments, `args`, are also arrays so that a
      class of similar problems can be solved together.
    * The size of the initial guesses, `x0`, is larger than O(100) elements.
      Otherwise, a naive loop may perform as well or better than a vector.
    Examples
    --------
    >>> from scipy import optimize
    >>> import matplotlib.pyplot as plt
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1
    ``fprime`` is not provided, use the secant method:
    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime_2=lambda x: 6 * x)
    >>> root
    1.0000000000000016
    Only ``fprime`` is provided, use the Newton-Raphson method:
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0
    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime_2=lambda x: 6 * x)
    >>> root
    1.0
    When we want to find zeros for a set of related starting values and/or
    function parameters, we can provide both of those as an array of inputs:
    >>> f = lambda x, a: x**3 - a
    >>> fder = lambda x, a: 3 * x**2
    >>> np.random.seed(4321)
    >>> x = np.random.randn(100)
    >>> a = np.arange(-50, 50)
    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ))
    The above is the equivalent of solving for each value in ``(x, a)``
    separately in a for-loop, just faster:
    >>> loop_res = [optimize.newton(f, x_0, fprime=fder, args=(a0,))
    ...             for x0, a0 in zip(x, a)]
    >>> np.allclose(vec_res, loop_res)
    True
    Plot the results found for all values of ``a``:
    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(a, analytical_result, 'o')
    >>> ax.plot(a, vec_res, '.')
    >>> ax.set_xlabel('$a$')
    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')
    >>> plt.show()
    """

    if tol <= 0.0:
        raise finpy_error("tol too small")

    max_iter = operator.index(max_iter)
    if max_iter < 1:
        raise finpy_error("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    p_0 = 1.0 * x_0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        for itr in range(max_iter):
            # first evaluate fval
            fval = func(p_0, args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return p_0
            fder = fprime(p_0, args)
            funcalls += 1

            # print("==>", itr, p0, fval, fder)

            if fder == 0:
                if disp is True:
                    print("Derivative is zero. Newton Failed to converge " +
                          "after ", str(itr + 1), "iterations, value is ", p_0)
                return None

            newton_step = fval / fder
            if fprime_2:
                fder2 = fprime_2(p_0, args)
                funcalls += 1
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p_0 - newton_step
            if np.isclose(p, p_0, rtol=rtol, atol=tol):
                return p
            p_0 = p
    else:
        # Secant method
        if x_1 is not None:
            if x_1 == x_0:
                raise ValueError("x1 and x0 must be different")
            p_1 = x_1
        else:
            eps = 1e-4
            p_1 = x_0 * (1 + eps)
            p_1 += (eps if p_1 >= 0 else -eps)
        q_0 = func(p_0, args)
        funcalls += 1
        q_1 = func(p_1, args)
        funcalls += 1
        if abs(q_1) < abs(q_0):
            p_0, p_1, q_0, q_1 = p_1, p_0, q_1, q_0
        for itr in range(max_iter):
            if q_1 == q_0:
                if p_1 != p_0:
                    if disp:
                        print("Tolerance reached. Failed to converge after ",
                              str(itr + 1), "iterations, value is ", str(p_1))
                    return None
                p = (p_1 + p_0) / 2.0
                return p
            else:
                if abs(q_1) > abs(q_0):
                    p = (-q_0 / q_1 * p_1 + p_0) / (1 - q_0 / q_1)
                else:
                    p = (-q_1 / q_0 * p_0 + p_1) / (1 - q_1 / q_0)
            if np.isclose(p, p_1, rtol=rtol, atol=tol):
                return p
            p_0, q_0 = p_1, q_1
            p_1 = p
            q_1 = func(p_1, *args)
            funcalls += 1

    if disp:
        print("Failed to converge after ", str(itr + 1),
              "iterations, value is ", str(p))

    return p


@njit(fastmath=True, cache=True)
def brent_max(func, lower_bound, upper_bound, args, xtol: float = 1e-5, max_iter: int = 500):
    """Uses a jitted version of the maximization routine from SciPy's fminbound.

    The algorithm is identical except that it's been switched to maximization
    rather than minimization, and the tests for convergence have been stripped
    out to allow for jit compilation.

    Note that the input function `func` must be jitted or the call will fail.

    Parameters
    ----------
    func : jitted function
    lower_bound : scalar for search
    upper_bound : scalar  for search
    args : tuple, optional
        Extra arguments passed to the objective function.
    max_iter : int, optional
        Maximum number of iterations to perform.
    xtol : float, optional
        Absolute error in solution `xopt` acceptable for convergence.

    Returns
    -------
    xf : float
        The maximizer
    fval : float
        The maximum value attained
    info : tuple
        A tuple of the form (status_flag, num_iter).  Here status_flag
        indicates whether or not the maximum number of function calls was
        attained.  A value of 0 implies that the maximum was not hit.
        The value `num_iter` is the number of function calls.

    Examples
    --------
    >>> @njit
    ... def f(x):
    ...     return -(x + 2.0)**2 + 1.0
    ...
    >>> xf, fval, info = brent_max(f, -2, 2)

    """
    if not np.isfinite(lower_bound):
        raise ValueError("a must be finite.")

    if not np.isfinite(upper_bound):
        raise ValueError("b must be finite.")

    if not lower_bound < upper_bound:
        raise ValueError("a must be less than b.")

    maxfun = max_iter
    status_flag = 0

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))

    fulc = lower_bound + golden_mean * (upper_bound - lower_bound)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = -func(x, *args)
    num = 1

    ffulc = fnfc = fx
    xm = 0.5 * (lower_bound + upper_bound)
    to_l1 = sqrt_eps * np.abs(xf) + xtol / 3.0
    to_l2 = 2.0 * to_l1

    while (np.abs(xf - xm) > (to_l2 - 0.5 * (upper_bound - lower_bound))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > to_l1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5 * q * r)) and (p > q * (lower_bound - xf)) and
                    (p < q * (upper_bound - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - lower_bound) < to_l2) or ((upper_bound - x) < to_l2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = to_l1 * si
            else:  # do a golden section step
                golden = 1

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = lower_bound - xf
            else:
                e = upper_bound - xf
            rat = golden_mean * e

        if rat == 0:
            si = np.sign(rat) + 1
        else:
            si = np.sign(rat)

        x = xf + si * np.maximum(np.abs(rat), to_l1)
        fu = -func(x, *args)
        num += 1

        if fu <= fx:
            if x >= xf:
                lower_bound = xf
            else:
                upper_bound = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                lower_bound = x
            else:
                upper_bound = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (lower_bound + upper_bound)
        to_l1 = sqrt_eps * np.abs(xf) + xtol / 3.0
        to_l2 = 2.0 * to_l1

        if num >= maxfun:
            status_flag = 1
            break

    fval = -fx
    info = status_flag, num

    return xf, fval, info


def bisection(func, x_1, x_2, args, xtol: float = 1e-6, max_iter: float = 100):
    """Bisection algorithm.

    You need to supply root brackets x1 and x2.
    TODO : @jit(fastmath=True, cache=True)
    """

    if np.abs(x_1 - x_2) < 1e-10:
        raise finpy_error("Brackets should not be equal")

    if x_1 > x_2:
        raise finpy_error("Bracket x2 should be greater than x1")

    f_1 = func(x_1, args)
    f_mid = func(x_2, args)

    if np.abs(f_1) < xtol:
        return x_1
    elif np.abs(f_mid) < xtol:
        return x_2

    if f_1 * f_mid >= 0:
        print("Root not bracketed")
        return None

    for i in range(0, max_iter):

        xmid = (x_1 + x_2) / 2.0
        f_mid = func(xmid, args)

        if f_1 * f_mid < 0:
            x_2 = xmid
        else:
            x_1 = xmid

        if np.abs(f_mid) < xtol:
            return xmid

    print("Bisection exceeded number of iterations", max_iter)
    return None


@njit(cache=True, fastmath=True)
def minimize_wolfe_powel(f, x, length, fargs=(), reduction=None, verbose=False, concise=False):
    """Minimize a differentiable multivariate function.

    credit : https://github.com/linesd/minimize/blob/master/optimizer/minimize.py

    The function uses conjugate gradients and approximate linesearches based
    On polynomial interpolation with Wolfe-Powel conditions

    Parameters
    ----------
    f : function to minimize. The function must return the value
        of the function (float) and a numpy array of partial
        derivatives of shape (D,) with respect to X, where D is
        the dimensionality of the function.
    x : numpy array - Shape : (D, 1)
        initial guess.
    length : int
        The length of the run. If positive, length gives the maximum
        number of line searches, if negative its absolute value gives
        the maximum number of function evaluations.
    args : tuple
        Tuple of parameters to be passed to the function f.
    reduction : float
        The expected reduction in the function value in the first
        linesearch (if None, defaults to 1.0)
    verbose : bool
        If True - prints the progress of minimize. (default is True)
    concise : bool
        If True - returns concise convergence info, only the minimium function
        value (necessary when optimizing a large number of parameters)
        (default is False)
    Return
    ------
    Xs : numpy array - Shape : (D, 1)
        The found solution.
    convergence : numpy array - Shape : (i, D+1)
        Convergence information. The first column is the function values
        returned by the function being minimized. The next D columns are
        the guesses of X during the minimization process.
        If concise = True, convergence information is only the minimum
        function value. This is necessary only when optimizing a large number
        of parameters.
    i : int
        Number of line searches or function evaluations depending on which
        was selected.
    The function returns when either its length is up, or if no further progress
     can be made (ie, we are at a (local) minimum, or so close that due to
     numerical problems, we cannot get any closer)
     Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).
     Converted to python by David Lines (2019-23-08)
    """
    _int = 0.1  # don't reevaluate within 0.1 of the limit of the current bracket
    _ext = 3.0  # extrapolate maximum 3 times the current step size
    max = 20  # max 20 function evaluations per line search
    RATIO = 10  # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2
    # SIG and RHO control the Wolfe-Powell conditions
    # SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the
    # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with RHO.

    # print("Minimizing %s ..." % f)

    if reduction is None:
        red = 1.0
    else:
        red = reduction

    S = 'Linesearch' if length > 0 else 'Function evaluation'

    i = 0  # run length counter
    ls_failed = 0  # no previous line search has failed
    f_0, df_0 = f(x, fargs)  # get initial function value and gradient
    df_0 = df_0.reshape(-1, 1)
    fX = []
    fX.append(f_0)
    x_d = []
    x_d.append(x)
    i += (length < 0)  # count epochs
    s = -df_0  # get column vec
    d_0 = -s.T @ s  # initial search direction (steepest) and slope
    x_3 = red / (1 - d_0)  # initial step is red/(|s|+1)

    while i < abs(length):  # while not finished
        i += (length > 0)  # count iterations

        X_0 = x
        dF_0 = df_0  # copy current vals
        M = max if length > 0 else min(max, -length - i)

        while 1:  # extrapolate as long as necessary
            x_2 = 0
            f_2 = f_0
            d_2 = d_0
            f_3 = f_0
            df_3 = df_0
            success = False

            while not success and M > 0:
                try:
                    M -= 1
                    i += (length < 0)  # count epochs
                    f_3, df_3 = f(x + x_3 * s, *list(*fargs))
                    df_3 = df_3.reshape(-1, 1)
                    if np.isnan(f_3) or np.isinf(f_3) or np.any(np.isnan(df_3) + np.isinf(df_3)):
                        raise Exception('Either nan or inf in function eval or gradients')
                    success = True
                except:  # catch any error occuring in f
                    x_3 = (x_2 + x_3) / 2  # bisect and try again

            if f_3 < F_0:
                X_0 = x + x_3 * s
                F_0 = f_3
                dF_0 = df_3  # keep best values

            d_3 = df_3.T @ s  # new slope
            if d_3 > SIG * d_0 or f_3 > f_0 + x_3 * RHO * d_0 or M == 0:
                break  # finished extrapolating

            x1 = x_2
            f1 = f_2
            d1 = d_2  # move point 2 to point 1
            x_2 = x_3
            f_2 = f_3
            d_2 = d_3  # move point 3 to point 2
            A = 6 * (f1 - f_2) + 3 * (d_2 + d1) * (x_2 - x1)  # make cubic extrapolation
            B = 3 * (f_2 - f1) - (2 * d1 + d_2) * (x_2 - x1)
            x_3 = x1 - d1 * (x_2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x_2 - x1)))  # num. error possible, ok!

            if np.iscomplex(x_3) or np.isnan(x_3) or np.isinf(x_3) or x_3 < 0:  # num prob | wrong sign
                x_3 = x_2 * _ext
            elif x_3 > x_2 * _ext:
                x_3 = x_2 * _ext
            elif x_3 < x_2 + _int * (x_2 - x1):
                x_3 = x_2 + _int * (x_2 - x1)

        while (abs(d_3) > -SIG * d_0 or f_3 > f_0 + x_3 * RHO * d_0) and M > 0:  # keep interpolating

            if d_3 > 0 or f_3 > f_0 + x_3 * RHO * d_0:  # choose subinterval
                x4 = x_3
                f4 = f_3
                d4 = d_3  # move point 3 to point 4
            else:
                x_2 = x_3
                f_2 = f_3
                d_2 = d_3  # move point 3 to point 2

            if f4 > f_0:
                x_3 = x_2 - (0.5 * d_2 * (x4 - x_2) ** 2) / (f4 - f_2 - d_2 * (x4 - x_2))  # quadratic interpolation
            else:
                A = 6 * (f_2 - f4) / (x4 - x_2) + 3 * (d4 + d_2)  # cubic interpolation
                B = 3 * (f4 - f_2) - (2 * d_2 + d4) * (x4 - x_2)
                x_3 = x_2 + (np.sqrt(B * B - A * d_2 * (x4 - x_2) ** 2) - B) / A  # num. error possible, ok!

            if np.isnan(x_3) or np.isinf(x_3):
                # if we had a numerical problem then bisect
                x_3 = (x_2 + x4) / 2

                # don't accept too close
            x_3 = max(min(x_3, x4 - _int * (x4 - x_2)), x_2 + _int * (x4 - x_2))
            f_3, df_3 = f(x + x_3 * s, *list(fargs))
            df_3 = df_3.reshape(-1, 1)

            if f_3 < F_0:
                X_0 = x + x_3 * s
                F_0 = f_3
                # keep best values
                dF_0 = df_3

            M -= 1
            # count epochs?!
            i += (length < 0)
            # new slope
            d_3 = df_3.T @ s

        if abs(d_3) < -SIG * d_0 and f_3 < f_0 + x_3 * RHO * d_0:  # if line search succeeded
            x = x + x_3 * s
            f_0 = f_3
            fX.append(f_0)
            x_d.append(x)  # update variables
            if verbose:
                print('%s %6i;  Value %4.6e\r' % (S, i, f_0))
            s = (df_3.T @ df_3 - df_0.T @ df_3) / (df_0.T @ df_0) * s - df_3  # Polack-Ribiere CG direction
            df_0 = df_3  # swap derivatives
            d_3 = d_0
            d_0 = df_0.T @ s
            if d_0 > 0:  # new slope must be negative
                s = -df_0.reshape(-1, 1);
                d_0 = -s.T @ s  # otherwise use steepest direction
            x_3 = x_3 * min(RATIO, d_3 / (d_0 - np.finfo(np.double).tiny))  # slope ratio but max RATIO
            ls_failed = False  # this line search did not fail
        else:
            x = X_0
            f_0 = F_0
            # restore best point so far
            df_0 = dF_0
            # line search failed twice in a row
            if ls_failed or i > abs(length):
                # or we ran out of time, so we give up
                break
            s = -df_0.reshape(-1, 1)
            # try steepest
            d_0 = -s.T @ s
            x_3 = 1 / (1 - d_0)
            # this line search failed
            ls_failed = True

    if concise:
        convergence = fX[-1]  # return only the minimum function value
    else:
        convergence = np.hstack((np.array(fX).reshape(-1, 1), np.array(x_d)[:, :, 0]))  # bundle convergence info

    return x, convergence, i
