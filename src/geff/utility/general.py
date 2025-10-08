"""
Some general purpose utility functions.
"""
import numpy as np
from typing import Callable

class AuxTol:
    """
    A simple wrapper which passes absolute and relative tolerance parameters.

    Wraps a function `f(*args, rtol, atol)` to give `f(*args, self.rtol, self.atol)` 


    """
    atol:float = 1e-20
    """Absolute tolerance"""
    rtol:float = 1e-6
    """Relative tolerance"""

    def __init__(self, func:Callable):
        self.func = func

    def __call__(self, *args):
        return self.func(*args, rtol=self.rtol, atol=self.atol)

@AuxTol
@np.vectorize
def heaviside(x : float, y:float, rtol:float, atol:float) -> float:
    r"""
    A smoothed version of the heaviside function.

    Returns 1 if x > y and 0 if x < y.
    
    The smoothness of the  transition is regulated by `max(abs(x)*rtol, atol))`.

    The function is wrapped by `AuxTol`, and will be updated according to the tolerances of a `GEFSolver`.

    It is vectorized using `numpy.vectorize`
    
    Parameters
    ----------
    x, y : floats
        the arguments of the heaviside function
    rtol, atol : floats
        tolerance parameters
    """
    eps = max(abs(x)*rtol, atol)
    arg = np.clip((x-y)/eps, -1e2, 1e18)
    return 1/(1+np.exp(-arg))