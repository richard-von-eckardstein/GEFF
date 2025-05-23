from src.BGQuantities.BGTypes import Val 
import numpy as np
from numpy.typing import NDArray
import math

def Friedmann(dphi : float|Val , V : float|Val, E : float|Val, B : float|Val,
               rhoChi : float|Val, H0 : float) -> float:
    """
    Calculate the Hubble rate from the Friedmann equation

    Parameters
    ----------
    dphi : float or Val
        the inflaton velocity, dphi/dt
    V : float or Val
        the inflaton potential energy, V(phi)
    E : float or Val
        the electric field expectation value E^2
    B : float or Val
        the magnetic field expectation value B^2
    rhoChi : float or Val
        the fermion energy density, rho_Chi
    H0 : float
        the Hubble rate at initialisation

    Returns
    -------
    float
        the Hubble rate, H
    """
    Hsq = (1/3) * (0.5 * dphi**2 + V + H0**2*( 0.5*(E + B) + rhoChi) ) 
    return np.sqrt(Hsq)

def EoMphi(dphi : float|Val, dV : float|Val, dI : float|Val, 
           G : float|Val, H : float|Val, H0 : float) -> float:
    """
    Calculate the inflaton acceleration from the Klein--Gordon equation

    Parameters
    ----------
    dphi : float or Val
        the inflaton velocity, dphi/dt
    dV : float or Val
        the inflaton potential gradient, dV/dphi
    dI : float or Val
        the inflaton--gauge-field coupling, dI/dphi
    G : float or Val
        the expectation value of -E.B
    H : float or Val
        the Hubble rate, H
    H0 : float
        the Hubble rate at initialisation
    

    Returns
    -------
    float
        the time derivative of dphi
    """
    return (-3*H * dphi - dV - dI*G*H0**2)

def EoMlnkh(kh : float|Val, dphi : float|Val, ddphi : float|Val,
             dI : float|Val, ddI : float|Val, xi : float|Val,
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
    xi : float or Val
        the instability parameter, xi
    a : float or Val
        the scale factor, a
    H : float or Val
        the Hubble rate, H

    Returns
    -------
    float
        the logarithmic time derivative of kh
    """
    
    r = 2*abs(xi)
        
    fc = a * H * r
    
    dHdt = 0. #approximation (quasi de Sitter)
    xiprime = (-dHdt * xi + ( ddI*dphi**2 + dI*ddphi)/2)/H
    rprime = 2*np.sign(xi)*xiprime
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def EoMF(F : NDArray, a : float|Val, kh : float|Val, sclrCpl : float,
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
        the inflaton gauge-field coupling, dI/dphi*dphi/dt = 2*H*xi
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

    bdrF = (
            dlnkhdt / (4*np.pi**2) 
            * (np.tensordot(np.ones_like(lams), W[:,0], axes=0)
                + np.tensordot(lams, W[:,1], axes=0))
            )

    dFdt = np.zeros_like(bdrF)

    dFdt[:-1,0] = (bdrF[:-1,0] - (4+ns[:-1])*dlnkhdt*FE[:-1]
                    - 2*scale*FG[1:] + 2*sclrCpl*FG[:-1])
    dFdt[:-1,1] = (bdrF[:-1,1] - (4+ns[:-1])*dlnkhdt*FB[:-1]
                    + 2*scale*FG[1:])
    dFdt[:-1,2] = (bdrF[:-1,2] - (4+ns[:-1])*dlnkhdt*FG[:-1]
                    + scale*(FE[1:] - FB[1:]) + sclrCpl*FB[:-1])

    #truncation conditions:
    ls = np.arange(1, L+1, 1)
    facl = np.array([math.comb(L, l) for l in range(1,L+1)])
    FEtr = np.sum( (-1)**(ls-1) * facl * FE[-2*ls], axis=0 )
    FBtr = np.sum( (-1)**(ls-1) * facl * FB[-2*ls], axis=0 )
    FGtr = np.sum( (-1)**(ls-1) * facl * FG[-2*ls], axis=0 )

    dFdt[-1,0] = (bdrF[-1,0] -  (4+ns[-1])*dlnkhdt*FE[-1]
                   - 2*scale*FGtr + 2*sclrCpl*FG[-1])
    dFdt[-1,1] = (bdrF[-1,1] - (4+ns[-1])*dlnkhdt*FB[-1]
                   + 2*scale*FGtr) 
    dFdt[-1,2] = (bdrF[-1,2] - (4+ns[-1])*dlnkhdt*FG[-1]
                   + scale*(FEtr - FBtr) + sclrCpl*FB[-1])

    return dFdt
