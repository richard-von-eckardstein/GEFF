import numpy as np
from numpy.typing import NDArray
import math
from GEFF.bgtypes import Val
from typing import Tuple

"""
Module for the equations of motions used for the model "SEOld", "SE_kh".

Functions
---------
Friedmann
    compute the Hubble rate
EoMPhi
    the Klein-Gordon equation in presence of gauge-field friction
EoMlnkh
    compute the time derivative of the instability scale kh
EoMF
    compute the time derivatives of the gauge-field bilinear tower
"""


def EoMlnkhSE(kh : float|Val, dphi : float|Val, ddphi : float|Val,
                dI : float|Val, ddI : float|Val,xieff : float|Val, s : float|Val,
                    a : float|Val, H : float|Val) -> float:
    """
    Calculate the logarithmic derivative of the instability scale kh.

    Parameters
    ----------
    kh : float or Val
        the instability scale kh
    dphi : float or Val
        the inflaton velocity, dphi/dt
    ddphi : float or Val
        the inflaton acceleration, d^2phi/d^2t
    dI : float or Val
        the inflaton--gauge-field coupling, dI/dphi
    ddI : float or Val
        derivative of the inflaton--gauge-field coupling, d^2I/d^2phi
    xieff : float or Val
        the effictive instability parameter, xi + sigmaB/(2H)
    s : float or val
        the effective electric conductivity, sigmaE/(2H)
    a : float or Val
        the scale factor, a
    H : float or Val
        the Hubble rate, H

    Returns
    -------
    float
        the logarithmic time derivative of kh
    """
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    
    fc = a * H * r

    #approximations
    dsigmaEdt = 0.
    dsigmaBdt = 0.
    dHdt = 0.#vals.vals["Hprime"]# #approximation  dHdt = alphaH**2  (slow-roll)

    xieffprime = (-dHdt * xieff + (ddI*dphi**2 + dI*ddphi + a*dsigmaBdt)/2)/H
    sEprime = (-dHdt * s + a*dsigmaEdt/2)/H
    rprime = ( (np.sign(xieff)+xieff/sqrtterm)*xieffprime
                 + sEprime*(s+1/2)/sqrtterm )
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def EoMDelta(delta : float|Val, sigmaE : float|Val) -> float:
    """
    Calculate the derivative of the cumulative electric damping.

    Parameters
    ----------
    delta : float or Val
        cumulative electric damping, delta
    sigmaE : float or Val
        electric conductivity, sigma_E

    Returns
    -------
    float
        the time derivative of delta
    """
    return -delta*sigmaE

def EoMrhoChi(rhoChi : float|Val, E : float|Val, G : float|Val,
               sigmaE : float|Val, sigmaB : float|Val, H : float|Val):
    """
    Calculate the derivative of the fermion energy density.

    Parameters
    ----------
    rhoChi : float or Val
        the fermion energy density, rhoChi
    E : float or Val
        the electric field expecation value, E^2
    G : float or Val
        the expectation value of -E.B
    sigmaE : float or Val
        electric conductivity, sigma_E
    sigmaB : float or Val
        magnetic conductivity, sigma_B
    H : float or Val
        the Hubble rate, H

    Returns
    -------
    float
        the time derivative of rhoChi
    """
    return (sigmaE*E - sigmaB*G - 4*H*rhoChi)


def EoMFSE(F : NDArray, a : float|Val, kh : float|Val, sclrCpl : float,
            sigmaE : float|Val, delta : float|Val, 
            W : NDArray, dlnkhdt : float, L : int=10) -> NDArray:
    """
    Calculate the derivative of the rescaled gauge-field bilinears.

    Parameters
    ----------
    F : NDArray
        A tower of gauge-bilinear quantities of shape (3,ntr)
    a : float or Val
        the scale factor, a
    kh : float or Val
        the instability scale kh
    sclrCpl : float or Val
        the effective inflaton gauge-field coupling, dI/dphi*dphi/dt + sigmaB = 2*H*xieff
    delta : float or Val
        cumulative electric damping, delta
    W : NDarray
        Whittaker functions used for boundary terms, shape (3,2)
    dlnkhdt : float
        the logarithmic derivative of the instability scale, dlnkh/dt
    L : int
        polynomial order for closing F(n+1) = Poly(L, F)

    Returns
    -------
    float
        the time derivative of F, shape (3,ntr)
    """
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    scale = kh/a

    W[2,1] = -W[2,1]

    ns = np.arange(0,  FE.shape[0])

    lams = (-1)**ns

    bdrF = (dlnkhdt*delta / (4*np.pi**2)
            * (np.tensordot(np.ones_like(lams), W[:,0], axes=0)
                + np.tensordot(lams, W[:,1], axes=0))
            )

    dFdt = np.zeros(bdrF.shape)

    dFdt[:-1,0] = (bdrF[:-1,0] - (4+ns[:-1])*dlnkhdt*FE[:-1]
                   - 2*sigmaE*FE[:-1] - 2*scale*FG[1:] + 2*sclrCpl*FG[:-1])
    dFdt[:-1,1] = (bdrF[:-1,1] - (4+ns[:-1])*dlnkhdt*FB[:-1]
                    + 2*scale*FG[1:])
    dFdt[:-1,2] = (bdrF[:-1,2] - (4+ns[:-1])*dlnkhdt*FG[:-1]
                   - sigmaE*FG[:-1] + scale*(FE[1:] - FB[1:]) + sclrCpl*FB[:-1])
    
    #truncation conditions:
    ls = np.arange(1, L+1, 1)
    facl = np.array([math.comb(L, j) for j in range(1,L+1)])
    FEtr = np.sum( (-1)**(ls-1) * facl * FE[-2*ls], axis=0 ) #-2*ls instead of -2*ls+1 since -1 is ntr not ntr-1
    FBtr = np.sum( (-1)**(ls-1) * facl * FB[-2*ls], axis=0 )
    FGtr = np.sum( (-1)**(ls-1) * facl * FG[-2*ls], axis=0 )

    #bilinears at truncation order ntr
    dFdt[-1,0] = (bdrF[-1,0] - (4+ns[-1])*dlnkhdt*FE[-1]
                   - 2*sigmaE*FE[-1] - 2*scale*FGtr + 2*sclrCpl*FG[-1])
    dFdt[-1,1] = (bdrF[-1,1] - (4+ns[-1])*dlnkhdt*FB[-1]
                   + 2*scale*FGtr) 
    dFdt[-1,2] = (bdrF[-1,2] - (4+ns[-1])*dlnkhdt*FG[-1]
                   - sigmaE*FG[-1] + scale*(FEtr - FBtr) + sclrCpl*FB[-1])

    return dFdt


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