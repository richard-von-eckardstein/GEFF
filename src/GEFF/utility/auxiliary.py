"""
Some general purpose utility functions.
"""
import numpy as np

class AuxTol:
    atol = 1e-20
    rtol = 1e-6

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self.func(*args, self.rtol, self.atol)

@AuxTol
@np.vectorize
def heaviside(x : float, y:float, rtol, atol) -> float:
    """
    A smoothed version of the heaviside function

    Parameters
    ----------
    x : float
        the argument of the heaviside function
    eps : float
        controls sharpness of the transition, smaller eps gives sharper transition

    Returns
    -------
    float
        heaviside(x)
    """
    eps = max(abs(x)*rtol, atol)
    arg = np.clip((x-y)/eps, -1e2, 1e18)
    return 1/(1+np.exp(-arg))