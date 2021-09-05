"""Copyright (C) Dominic O'Kane.

Suppress error traceback messages in Jupyter Notebook

iPython dependency is only loaded if required.
"""

import traceback
import sys

ipython = None
try:
    from IPython import get_ipython
    ipython = get_ipython()
except:
    pass


def _hide_traceback():
    etype, value, _ = sys.exc_info()
    if ipython is not None:
        msg = ipython._showtraceback(
            etype,
            value,
            ipython.InteractiveTB.get_exception_only(etype, value)
        )
    else:
        msg = None
    return msg


def func_name():
    """Returns the function name"""
    return traceback.extract_stack(None, 2)[0][2]


def suppress_traceback():
    """Suppress Tracebacks"""

    sys.tracebacklimit = 0
    ipython.showtraceback = _hide_traceback


class finpy_error(Exception):
    """Simple error class specific to FinPy. Need to decide how to handle
    FinancePy errors. Work in progress."""

    def __init__(self, message: str):
        """Create FinError object by passing a message string."""
        self._message = message

    def _print(self):
        """Allows to print"""
        print("FinError:", self._message)
