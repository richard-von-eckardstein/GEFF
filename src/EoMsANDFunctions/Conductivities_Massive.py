import numpy as np
from src.BGQuantities.BGTypes import Val 
from typing import Tuple

"""
Module for computing electric and magnetic conductivities used in "SEOld", "SE_kh".

Functions
---------
ComputeSigmaCollinear
    Compute electric & magnetic conductivities and the damping scale kS
    assuming collinear electric and magnetic fields (picture: electric, magnetic)

ComputeImprovedSigma
    Compute electric & magnetic conductivities and the damping scale kS
    using boosts to the collinear fram (picture: mixed).
"""

def ComputeSigmaCollinear(a  : float|Val, H  : float|Val,
                           E  : float|Val, B : float|Val, G : float|Val,
                             pic : int, H0 : float, mF : float, geff : float ) -> Tuple[float, float, float]:
    """
    Compute electric & magnetic conductivities and the damping scale kS
    assuming collinear electric and magnetic fields (picture: electric, magnetic).

    Parameters
    ----------
    a : float or Val
        the scale factor, a
    H : float or Val
        the Hubble rate, H
    E : float or Val
        the electric field expectation value E^2
    B : float or Val
        the magnetic field expectation value B^2
    G : float or Val
        the expectation value of -E.B
    pic : int
        an integer specifying electric (pic=-1) or magnetic pictures (pic=1) 
    H0 : float
        the Hubble rate at initialisation

    Returns
    -------
    float
        the electric damping, sigmaE
    float
        the magnetic damping, sigmaB
    float
        the damping scale, kS
    """     
    #geff = C * e^3
    mu = (E+B)
    if mu<=0:       
        return 0., 0., 0.
    else:
        mFbar = mF / H0

        sigma = (geff**3)/(6.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B/E)))
        sigmaE =  np.sqrt(B) * (min(1., (1.- pic))*E + max(-pic, 0.)*B) * sigma / (E+B) * np.exp((-np.pi*mFbar**2)/(geff*np.sqrt(E)))           #cdv glaube so war richtig sqrt(E), E= E^2
        sigmaB = -np.sign(G) * np.sqrt(E)*(min(1., (1.+ pic))*B + max(pic,0.)*E)* sigma/(E+B) * np.exp((-np.pi*mFbar**2)/(geff*np.sqrt(E)))
        

        ks = geff*E**(1/4)*a

        return sigmaE, sigmaB, ks

def ComputeImprovedSigma(a  : float|Val, H  : float|Val,
                           E  : float|Val, B : float|Val, G : float|Val,
                            H0 : float, mF : float, geff : float) -> Tuple[float, float, float]:
    """
    Compute electric & magnetic conductivities and the damping scale kS
    using boosts to the collinear fram (picture: mixed).

    Parameters
    ----------
    a : float or Val
        the scale factor, a
    H : float or Val
        the Hubble rate, H
    E : float or Val
        the electric field expectation value E^2
    B : float or Val
        the magnetic field expectation value B^2
    G : float or Val
        the expectation value of -E.B
    H0 : float
        the Hubble rate at initialisation

    Returns
    -------
    float
        the electric damping, sigmaE
    float
        the magnetic damping, sigmaB
    float
        the damping scale, kS
    """  
    Sigma = np.sqrt((E - B)**2 + 4*G**2)
    if Sigma<=0:
        return 0., 0., 0.

    #if np.any(Sigma <= 0):
    #    return 0., 0., 0.       # chatgpt cdv: damit nicht mehr ValueError, didnt work

    else:
        mFbar = mF / H0

        Eprime = np.sqrt((E - B + Sigma)/2)
        Bprime = np.sqrt((B- E + Sigma)/2)
        Sum = E + B + Sigma
        
        sigma = np.sqrt(2)*(geff**3/(6.*np.pi**2) / (np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime))) * np.exp((-np.pi*mFbar**2)/(geff*Eprime))
        
        ks = geff*Eprime**(1/2)*a

        return abs(G)*Eprime*sigma, -G*Bprime*sigma, ks