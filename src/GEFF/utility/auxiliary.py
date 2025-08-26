"""
Some general purpose utility functions.
"""
import numpy as np

def heaviside(x : float, eps : float) -> float:
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
    return 1/(1+np.exp(-x/eps))