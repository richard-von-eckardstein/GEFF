import numpy as np
from GEFF.BGTypes import Val 
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
                             pic : int, H0 : float) -> Tuple[float, float, float]:
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
    mu = (E+B)
    if mu<=0:
        return 0., 0., 0.
    else:
        mu = (mu/2)**(1/4)
        mz = 91.2/(2.43536e18)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*H0))))
        
        C = 41/12

        sigma = (C*gmu**3/(6*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B/E))))
        sigmaE =  np.sqrt(B) * (min(1., (1.- pic))*E + max(-pic, 0.)*B) * sigma / (E+B)         
        sigmaB = -np.sign(G) * np.sqrt(E)*(min(1., (1.+ pic))*B + max(pic,0.)*E)* sigma/(E+B)
        
        ks = C**(1/3)*gmu*E**(1/4)*a
        
        return sigmaE, sigmaB, ks

def ComputeImprovedSigma(a  : float|Val, H  : float|Val,
                           E  : float|Val, B : float|Val, G : float|Val,
                            H0 : float) -> Tuple[float, float, float]:
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
    else:
        mz = 91.2/(2.43536e18)
        mu = ((Sigma)/2)**(1/4)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*H0))))

        Eprime = np.sqrt( (E - B + Sigma)/2 )
        Bprime = np.sqrt( (B- E + Sigma)/2 )
        Sum = E + B + Sigma

        C = 41/12
        
        sigma = ( C*gmu**3/(6*np.pi**2) / (np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime)))
        
        ks = C**(1/3)*gmu*Eprime**(1/2)*a

        return 2**(1/2)*abs(G)*Eprime*sigma, -2**(1/2)*G*Bprime*sigma, ks