import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

"""
Module providing mode equations for gauge-field modes and functions used for
initialising in guage-field modes in Bunch-Davies

Functions
---------
BDClassic
    Returns gauge-field modes in conventional Bunch-Davies vaccum.

ModeEoMClassic
    guage-field mode equation for a axion--vector coupling
    used in the model "Classic".

BDDamped
    Returns gauge-field modes in damped Bunch-Davies vaccum.

ModeEoMSchwinger
    guage-field mode equation in the presence of a damping medium,
    used in the model "SEOld".

ModeEoMSchwinger_kS
    guage-field mode equation in the presence of a damping medium,
    accounts for a characterstic damping scale
    used in the model "SE_kh".

"""

def BDClassic(t: float, k : float) -> NDArray:
    """
    Returns gauge-field modes in conventional Bunch-Davies vaccum.

    Parameters
    ----------
    t : float
        time if initilisation
    k : float
        comoving wavenumber in Hubble units

    Returns
    -------
    y : NDarray
        an array re-scaled gauge-field modes in Bunch-Davies, shape (8)

    """
    return np.array([1., 0, 0, -1., 1, 0, 0, -1.])

def ModeEoMClassic(t: float, y : NDArray, k : float, a : CubicSpline,
                    xi : CubicSpline, H : CubicSpline) -> NDArray:
    """
    guage-field mode equation for a axion--vector coupling
    used in the model "Classic".

    Parameters
    ----------
    t : float
        time
    y : NDArray
        gauge-field mode and its derivatives for both helicities +/-.
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, a*deta = dt
    k : float
        comoving wavenumber in Hubble units
    a : CubicSpline
        scale factor as function of time
    xi : CubicSpline
        instability scale as function of time
    H : CubicSpline
        Hubble rate as function of time
        
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

def BDDamped(t : float, k : float, a : CubicSpline, delta : CubicSpline,
              sigmaE : CubicSpline) -> NDArray:
    """
    Returns gauge-field modes in damped Bunch-Davies vaccum.

    Parameters
    ----------
    t : float
        time if initilisation
    k : float
        comoving wavenumber in Hubble units
    a : CubicSpline
        scale factor as function of time
    sigmaE : CubicSpline
        electric damping as function of time
    delta : CubicSpline
        cumulative electric damping as function of time
    Returns
    -------
    y : NDarray
        an array re-scaled gauge-field modes in Bunch-Davies, shape (8)

    """
    yini = np.array([1., -1/2*sigmaE(t)*a(t)/k, 0, -1.,
                     1., -1/2*sigmaE(t)*a(t)/k, 0, -1.])*np.sqrt( delta(t) )
    return yini


def ModeEoMSchwinger(t: float, y : NDArray, k : float,
                    a : CubicSpline, xieff : CubicSpline, H : CubicSpline,
                    sigmaE : CubicSpline) -> NDArray:
    """
    guage-field mode equation in the presence of a damping medium,
    used in the model "SEOld".

    Parameters
    ----------
    t : float
        time
    y : NDArray
        gauge-field mode and its derivatives for both helicities +/-.
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, a*deta = dt
    k : float
        comoving wavenumber in Hubble units
    a : CubicSpline
        scale factor as function of time
    xieff : CubicSpline
        effective instability scale as function of time
    H : CubicSpline
        Hubble rate as function of time
    sigmaE : CubicSpline
        electric damping as function of time
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y
    """
    
    dydt = np.zeros_like(y)
    
    drag = sigmaE(t)
    dis1 = k / a(t)
    dis2 = 2*H(t)*xieff(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = - drag * y[1] - (dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = - drag * y[3] - (dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = - drag * y[5] - (dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = - drag * y[7] - (dis1  - lam * dis2) * y[6]
    
    return dydt

def ModeEoMSchwinger_kS(t: float, y : NDArray, k : float,
                    a : CubicSpline, xi : CubicSpline, H : CubicSpline,
                    sigmaE : CubicSpline, sigmaB : CubicSpline,
                      kS : CubicSpline) -> NDArray:
    """
    guage-field mode equation in the presence of a damping medium,
    accounts for a characterstic damping scale
    used in the model "SE_kh".

    Parameters
    ----------
    t : float
        time
    y : NDArray
        gauge-field mode and its derivatives for both helicities +/-.
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, a*deta = dt
    k : float
        comoving wavenumber in Hubble units
    a : CubicSpline
        scale factor as function of time
    xieff : CubicSpline
        effective instability scale as function of time
    H : CubicSpline
        Hubble rate as function of time
    sigmaE : CubicSpline
        electric damping as function of time
    kS : CubicSpline
        damping scale above which modes are not damped
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y
    """
    dydt = np.zeros_like(y)

    theta = np.heaviside(kS(t) - k, 0.5)
    
    drag = theta*sigmaE(t)
    dis1 = k / a(t)
    dis2 = 2*H(t)*xi(t) + theta*sigmaB(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = - drag * y[1] - (dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = - drag * y[3] - (dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = - drag * y[5] - (dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = - drag * y[7] - (dis1  - lam * dis2) * y[6]
    
    return dydt



