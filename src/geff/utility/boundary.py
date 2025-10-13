r"""
This module facilitates computing boundary terms which appear when computing the evolution of gauge-field bilinears like $\langle \boldsymbol{E}^2 \rangle$
due to the time dependence of the UV-regulator scale $k_{\rm h}$.

All functions in this module return the same quantities, namely
$$E_\lambda(\xi, s) = \frac{1}{r^2}\left| (i r - i \lambda \xi - s) W_{-i \lambda \xi, \frac{1}{2} + s}(-2 i r) + W_{1-i \lambda \xi, 1/2 + s}(-2 i r) \right|^2 \, , $$
$$B_\lambda(\xi, s) = \left| W_{-i \lambda \xi, \frac{1}{2} + s}(-2 i r) \right|^2 \, , $$
$$G_\lambda(\xi, s) = \frac{1}{r}\left[\operatorname{Re}\left[W_{1-i \lambda \xi, 1/2 + s}(-2 i r) W_{i \lambda \xi, \frac{1}{2} + s}(2 i r)\right] -  s \left| W_{-i \lambda \xi, \frac{1}{2} + s}(-2 i r)\right|^2 \right]\, ,$$
with $r = |\xi| + \sqrt{\xi^2 + s^2 + s}$, the Whittaker-W function $W_{\kappa, \mu}(x)$,the instability parameter  $\xi$ , and $s= \sigma_{\rm E}/(2H)$ an effective electric conductivity.

The functions in this module return an array of shape (3,2), with the first index corresponding to $E$, $B$, $G$ and the second index to helicity $\lambda=\pm 1$.
"""
import numpy as np
from scipy.special import gamma
from mpmath import whitw, mp
from functools import lru_cache

#set accuracy of mpmath
mp.dps = 8

def boundary_pai(xi) -> np.ndarray:
    """
    Compute boundary terms in the absence of Schwinger pair production.
    """
    if abs(xi) >= 3:
        return boundary_approx_pai(xi)
    else:
        return boundary_exact(xi, 0)
    
def boundary_fai(xi, s) ->  np.ndarray:
    """
    Compute boundary terms including Schwinger pair production.
    """
    if abs(xi) >= 4:
        return boundary_approx_fai(xi, s)
    else:
        return boundary_exact(xi, s)



@lru_cache(maxsize=int(1e6))
def whittaker_w(xi, s):
    r = (abs(xi) + np.sqrt(xi**2 + s**2 + s))
    w = whitw(-xi*1j, 1/2 + s, -2j*r)
    w1 = whitw(1-xi*1j, 1/2 + s, -2j*r)
    return complex(w), complex(w1)
    

def boundary_exact(xi : float, s : float) -> np.ndarray:
    """
    Compute boundary terms using exact Whittaker functions.

    Parameters
    ----------
    xi : float
    s : float

    Returns
    -------
    W : NDArray
    """
    xi = float(xi)
    wp, w1p = whittaker_w(xi, s)
    wm, w1m = whittaker_w(-xi, s)

    w = np.array([wp, wm])
    w1 = np.array([w1p, w1m])

    lam = np.array([1, -1])

    expterm = np.exp(np.pi*xi*lam)

    r = (abs(xi) + np.sqrt(xi**2 + s**2 + s))
    
    Fterm = np.zeros((3, 2))

    Fterm[0,:] = expterm*abs((1j*r - 1j**lam*xi -s) * w + w1)**2/r**2
    Fterm[1,:] = expterm*abs(w)**2
    Fterm[2,:] = expterm*((w1*w.conjugate()).real - s*abs(w)**2)/r

    return Fterm

def boundary_approx_pai(xi : float) -> np.ndarray:
    r"""
    Use approximate formulas to compute boundary terms for $|\xi| > 3$ and $s=0$.

    The approximations are taken from the Appendix B in [2109.01651](https://arxiv.org/abs/2109.01651).

    Parameters
    ----------
    xi : float

    Returns
    -------
    W : NDArray
    """
    if (abs(xi) >= 3):
        Fterm = np.zeros((3, 2))
        sgnsort = int((1-np.sign(xi))/2)

        xi = abs(xi)
        g1 = gamma(2/3)**2
        g2 = gamma(1/3)**2
        t1 = (3/2)**(1/3)*g1/(np.pi*xi**(1/3))
        t2 = -np.sqrt(3)/(15*xi)
        t3 = (2/3)**(1/3)*g2/(100*np.pi*xi**(5/3))
        t4 = (3/2)**(1/3)*g1/(1575*np.pi*xi**(7/3))
        t5 = -27*np.sqrt(3)/(19250*xi**3)
        t6 = 359*(2/3)**(1/3)*g2/(866250*np.pi*xi**(11/3))
        t7 = 8209*(3/2)**(1/3)*g1/(13162500*np.pi*xi**(13/3))
        t8 = -690978*np.sqrt(3)/(1861234375*xi**5)
        t9 = 13943074*(2/3)**(1/3)*g2/(127566140625*np.pi*xi**(17/3))
        Fterm[0, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9

        t1 = 1
        t2 = -9/(2**(10)*xi**2)
        t3 = 2059/(2**(21)*xi**4)
        t4 = -448157/(2**31*xi**6)
        Fterm[0, 1-sgnsort] = np.sqrt(2)*(t1 + t2 + t3 + t4)

        t1 = (2/3)**(1/3)*g2*xi**(1/3)/(np.pi)
        t2 = 2*np.sqrt(3)/(35*xi)
        t3 = -4*(2/3)**(1/3)*g2/(225*np.pi*xi**(5/3))
        t4 = 9*(3/2)**(1/3)*g1/(1225*np.pi*xi**(7/3))
        t5 = 132*np.sqrt(3)/(56875*xi**3)
        t6 = -9511*(2/3)**(1/3)*g2/(5457375*np.pi*xi**(11/3))
        t7 = 1448*(3/2)**(1/3)*g1/(1990625*np.pi*xi**(13/3))
        t8 = 1187163*np.sqrt(3)/(1323765625*xi**5)
        t9 = -22862986*(2/3)**(1/3)*g2/(28465171875*np.pi*xi**(17/3))
        Fterm[1, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9

        t1 = 1
        t2 = 11/(2**(10)*xi**2)
        t3 = -2397/(2**(21)*xi**4)
        t4 = 508063/(2**31*xi**6)
        Fterm[1, 1-sgnsort] = 1/np.sqrt(2)*(t1 + t2 + t3 + t4)

        t1 = 1/np.sqrt(3)
        t2 = -(2/3)**(1/3)*g2/(10*np.pi*xi**(2/3))
        t3 = 3*(3/2)**(1/3)*g1/(35*np.pi*xi**(4/3))
        t4 = -np.sqrt(3)/(175*xi**2)
        t5 = -41*(2/3)**(1/3)*g2/(34650*np.pi*xi**(8/3))
        t6 = 10201*(3/2)**(1/3)*g1/(2388750*np.pi*xi**(10/3))
        t7 = -8787*np.sqrt(3)/(21896875*xi**4)
        t8 = -1927529*(2/3)**(1/3)*g2/(4638768750*np.pi*xi**(14/3))
        t9 = 585443081*(3/2)**(1/3)*g1/(393158390625*np.pi*xi**(16/3))
        t10 = -65977497*np.sqrt(3)/(495088343750*xi**6)
        Fterm[2, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9+t10

        t1 = 1
        t2 = -67/(2**(10)*xi**2)
        t3 = 21543/(2**(21)*xi**4)
        t4 = -6003491/(2**31*xi**6)
        Fterm[2, 1-sgnsort] = -np.sqrt(2)/(32*xi)*(t1 + t2 + t3 + t4) 
        return Fterm
    else:
        raise ValueError("abs(xi) needs to be larger than three for this approximation.")

def boundary_approx_fai(xi :float, s : float) -> np.ndarray:
    r"""
    Use approximate formulas to compute boundary terms for  $|\xi| > 4$ and arbitrary $s$.

    The approximations are taken from the Appendix B in [2109.01651](https://arxiv.org/abs/2109.01651).

    Parameters
    ----------
    xi : float

    Returns
    -------
    W : NDArray
    """
    if (abs(xi) >= 4):
        Fterm = np.zeros((3, 2))
        sgnsort = int((1-np.sign(xi))/2)
        
        xi = abs(xi)
        r = xi + np.sqrt(xi**2 + s**2 + s)
        psi = 2*np.sqrt(xi**2 + s**2 + s)/r
        rpsi = (psi/r**2)**(1/3)
        spsi = 5*s/psi**(2/3)
        
        g1 = gamma(2/3)**2
        g2 = gamma(1/3)**2
        
        t1 = 3**(1/3)*g1/np.pi
        t2 = -2/(5*np.sqrt(3))*(1+spsi)
        t3 = g2/(3**(1/3)*25*np.pi)*(1+spsi)**2
        t4 = 3**(1/3)*4*g1/(1575)*(1-27*spsi)
        t5 = 4*np.sqrt(3)/(875)*(-27/11+2*spsi + spsi**2)
        Fterm[0, sgnsort] = (psi/r)**(1/3)*(t1 + t2*rpsi + t3*rpsi**2 + t4*rpsi**3 + t5*rpsi**4)

        t1 = 1
        t2 = s/(16*xi**2)*(3*xi-r)/r
        t3 = s**2/(4*xi*r)
        Fterm[0, 1-sgnsort] = 2*np.sqrt(xi/r)*(t1 + t2 + t3)

        t1 = g2/(3**(1/3)*np.pi)
        t2 = 4*np.sqrt(3)/35
        t3 = -16*g2/(3**(1/3)*225*np.pi)
        Fterm[1, sgnsort] = (r/psi)**(1/3)*(t1 + t2*rpsi**2 + t3*rpsi**3)

        Fterm[1, 1-sgnsort] = 0.5*(r/xi)**(1/2)

        t1 = 1/np.sqrt(3)
        t2 = -g2/(3**(1/3)*5*np.pi)*(1+spsi)
        t3 = 3**(1/3)*6*g1/(35*np.pi)
        t4 = -4*np.sqrt(3)/(175)*(1+spsi)
        Fterm[2, sgnsort] = t1 + t2*rpsi + t3*rpsi**2 + t4*rpsi**3

        Fterm[2, 1-sgnsort] = -((3*xi -r)/xi + 8*s)/(16*np.sqrt(xi*r))
        return Fterm
    else:
        raise ValueError("abs(xi) needs to be larger than four for this approximation.")

