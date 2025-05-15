import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

def BDClassic(t: float, k : float):
    return np.array([1., 0, 0, -1., 1, 0, 0, -1.])

def ModeEoMClassic(t: float, y : ArrayLike, k : float, a : CubicSpline, xi : CubicSpline, H : CubicSpline):
    """
    Compute the time derivative of the gauge-field mode and its derivatives for a fixed comoving wavenumber at a given moment of time t

    Parameters
    ----------
    y : numpy.array
        contains the gauge-field mode and its derivatives for both helicities +/- (in Hubble units).
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, a*deta = dt
    k : float
        the comoving wavenumber in Hubble units
    a : CubicSpline
        the scale factor interpolated over time t
    xi : float
        the instability parameter interpolated over time t
    H : float
        the Hubble rate interpolated over time t
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y

    """
    
    dydt = np.zeros_like(y)
    
    dis1 = k / a(t)
    dis2 = 2*H(t)*xi(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = -(dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = -(dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = -(dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = -(dis1  - lam * dis2) * y[6]
    
    return dydt


