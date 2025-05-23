import numpy as np
import math
from mpmath import whitw, whitm, mp
from numpy.typing import NDArray

"""
Module for computing boundary terms based on
whittaker functions W(-i*lambda*xi, 1/2+s, 2|xi|), lambda=+/-

Functions
---------
WhittakerExact
    compute boundary terms using exact whittaker functions
WhittakerApprox
    use approximate formulas for WhittakerExact(xi, 0) when applicable.
WhittakerApprox_SE
    use approximate formulas for WhittakerExact(xi, s) when applicable.
"""

mp.dps = 8

def WhittakerExact(xi : float, s : float) -> NDArray:
    """
    compute boundary terms using exact whittaker functions

    Parameters
    ----------
    xi : float
    s : float

    Returns
    -------
    NDArray
        Boundary terms for gauge bilinears, shape (3,2)
    """
    r = (abs(xi) + np.sqrt(xi**2 + s**2 + s))
    
    Whitt1Plus = whitw(-xi*(1j), 1/2 + s, -2j*r)
    Whitt2Plus = whitw(1-xi*(1j), 1/2 + s, -2j*r)

    Whitt1Minus = whitw(xi*(1j), 1/2 + s, -2j*r)
    Whitt2Minus = whitw(1+xi*(1j), 1/2 + s, -2j*r)
        
    exptermPlus = np.exp(np.pi*xi)
    exptermMinus = np.exp(-np.pi*xi)
    
    Fterm = np.zeros((3, 2))

    Fterm[0,0] = exptermPlus*abs((1j*r - 1j*xi -s) * Whitt1Plus + Whitt2Plus)**2/r**2
    Fterm[0,1] = exptermMinus*abs((1j*r + 1j*xi -s) * Whitt1Minus + Whitt2Minus)**2/r**2

    Fterm[1,0] = exptermPlus*abs(Whitt1Plus)**2
    Fterm[1,1] = exptermMinus*abs(Whitt1Minus)**2

    Fterm[2,0] = exptermPlus*((Whitt2Plus*Whitt1Plus.conjugate()).real - s * abs(Whitt1Plus)**2)/r
    Fterm[2,1] = exptermMinus*((Whitt2Minus*Whitt1Minus.conjugate()).real - s * abs(Whitt1Minus)**2)/r

    return Fterm

#Whittaker Functions
def WhittakerApprox(xi : float):
    """
    use approximate formulas for WhittakerExact(xi, 0) where applicable.

    Parameters
    ----------
    xi : float

    Returns
    -------
    NDArray
        Boundary terms for gauge bilinears, shape (3,2)
    """
    if (abs(xi) >= 3):
        Fterm = np.zeros((3, 2))
        sgnsort = int((1-np.sign(xi))/2)

        xi = abs(xi)
        g1 = math.gamma(2/3)**2
        g2 = math.gamma(1/3)**2
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
    else:
        Fterm = WhittakerExact(xi, 0.)
    return Fterm

def WhittakerApproxSE(xi :float, s : float):
    """
    use approximate formulas for WhittakerExact(xi, s) where applicable.

    Parameters
    ----------
    xi : float

    Returns
    -------
    NDArray
        Boundary terms for gauge bilinears, shape (3,2)
    """
    if (abs(xi) >= 4):
        Fterm = np.zeros((3, 2))
        sgnsort = int((1-np.sign(xi))/2)
        
        xi = abs(xi)
        r = xi + np.sqrt(xi**2 + s**2 + s)
        psi = 2*np.sqrt(xi**2 + s**2 + s)/r
        rpsi = (psi/r**2)**(1/3)
        spsi = 5*s/psi**(2/3)
        
        g1 = math.gamma(2/3)**2
        g2 = math.gamma(1/3)**2
        
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
    else:
        Fterm = WhittakerExact(xi, s)
    return Fterm
