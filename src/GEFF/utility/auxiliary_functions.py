import numpy as np

"""
Module for useful auxiliary functions used in many model-files.

Functions
---------
Heaviside
    a smoothed version of the heaviside function 
"""

def Heaviside(x : float, eps : float) -> float:
    """
    A smoothed version of the heaviside function

    Parameters
    ----------
    dx : float
        the argument of the heaviside function
    eps : float
        controls sharpness of the transition, smaller eps gives sharper transition

    Returns
    -------
    float
        Heaviside(x)
    """
    return 1/(1+np.exp(-x/eps))