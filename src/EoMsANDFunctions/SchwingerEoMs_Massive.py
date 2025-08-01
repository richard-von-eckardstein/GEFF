import numpy as np
from numpy.typing import NDArray
import math
from src.BGQuantities.BGTypes import Val

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


# inlcude variable EoS cdv
def EoMrhoChi(rhoChi : float|Val, E : float|Val, G : float|Val, 
               sigmaE : float|Val, sigmaB : float|Val, H : float|Val, H0 : float, mF : float, geff : float):      
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
    mFbar = mF/H0
    w = 1/3 * E * (geff / H)**2 / (mFbar**2 + E * (geff / H)**2)  #frage mit einheiten? E = (E-Field)**2

    return (sigmaE*E - sigmaB*G - 3*H*(1+w)*rhoChi)


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
    facl = np.array([math.comb(L, l) for l in range(1,L+1)])
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


