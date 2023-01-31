import numpy as np
from mpmath import whitw
import math

alpha=0

def potential(phi):
    #potential energy as function of field amplitude
    return 0.5*phi**2 * M**2

def dVdphi(phi):
    #derivative of potential energy as function of field amplitude
    return phi * M**2 

def dIdphi(phi):
    #derivative of gauge field coupling term as function of field amplitude
    return beta/Mpl

def ddIddphi(phi):
    #second derivative of gauge field coupling term as function of field amplitude
    return 0

def GetXi(dphidt, Iterm, H):
    #Compute Xi for a given dIdphi, dphidt and H
    return Iterm * dphidt / (2*H)

def GetS(a, H, sigma):
    #Compute s for a given a, H and sigma
    return a**(alpha) * sigma / (2*H)

def FriedmannEq(a, dphidt, V, E, B):
    #E: EE
    #B: BB
    #sc[0]: phi
    #sc[1]: dphidt
    
    dadt = a*np.sqrt( (1/(3*Mpl**2)) * (0.5 * dphidt**2 + a**(2*alpha) * V + 0.5*a**(2*alpha) * (E+B)) )
    
    return dadt

def EoMphi(dphidt, Vterm, Iterm, G, a, H):
    #G: -1/2(EB + BE)
    #sc[0]: phi
    #sc[1]: dphidt
    
    dscdt = np.zeros(2)
    
    dscdt[0] = dphidt
    dscdt[1] = (alpha-3)*H*dphidt - a**(2*alpha)*Vterm - a**(2*alpha)*Iterm*G
    
    return dscdt

def BoundaryComputations(kh, dphidt, ddphiddt, Iterm, I2term, a, H, ntr, sigma=0, dsigmadt=0):
    xi = GetXi(dphidt, Iterm, H)
    s = GetS(a, H, sigma)
    r = (abs(xi) + np.sqrt(xi**2 + s**2 + s))
    f = a**(1-alpha) * H * (r)
    """fprime =  a*(1-alpha) * ( (1-alpha) * H**2 * f + 
                            0.5 * ( (I2term * dphidt**2 + Iterm*ddphiddt) * g(Iterm*dphidt) 
                                    + 1/np.sqrt(Iterm**2 * dphidt**2 + a**(2*alpha)*(sigma**2 + 2*H*sigma)) *
                                  (Iterm*I2term*dphidt**3 + Iterm**2*dphidt*ddphiddt 
                                   + a**(2*alpha) * (sigma * (alpha*H*sigma + dsigmadt) 
                                                     + a**(-alpha)*H * (2*alpha*sigma + dsigmadt))) ))"""
    
    fprime = Iterm*g(dphidt)*ddphiddt*a + abs(dphidt)*a*H*Iterm
    
    if (fprime > 0):
        kh = f
        dlnkhdt = fprime/kh
    else:
        dlnkhdt = 0
    
    bdrF = ComputeBoundary(a, kh, dlnkhdt, ntr, r, xi, s)
    
    return kh, dlnkhdt, bdrF
    
def ComputeBoundary(a, kh, dlnkhdt, ntr, r, xi, s, approx=False):
    
    if (s==0):
        delta = 1
    else:
        #left to be implemented
        delta = 1
        
    prefac = dlnkhdt * delta/ (4*np.pi**2)
    
    if (approx):
        EtermPlus = approxPosE(xi)
        EtermMinus = approxMinE(xi)
        
        BtermPlus = approxPosB(xi)
        BtermMinus = approxMinB(xi)
        
        GtermPlus = approxPosG(xi)
        GtermMinus = approxMinG(xi)
        
    else:
        Whitt1Plus = whitw(-xi*(1j), 1/2 + s, -2j*r)
        Whitt2Plus = whitw(1-xi*(1j), 1/2 + s, -2j*r)

        Whitt1Minus = whitw(xi*(1j), 1/2 + s, -2j*r)
        Whitt2Minus = whitw(1+xi*(1j), 1/2 + s, -2j*r)

        exptermPlus = np.exp(np.pi*xi)
        exptermMinus = np.exp(-np.pi*xi)

        EtermPlus = exptermPlus*abs((1j*r - 1j*xi -s) * Whitt1Plus + Whitt2Plus)**2/r**2
        EtermMinus = exptermMinus*abs((1j*r + 1j*xi -s) * Whitt1Minus + Whitt2Minus)**2/r**2
        
        BtermPlus = exptermPlus*abs(Whitt1Plus)**2
        BtermMinus = exptermMinus*abs(Whitt1Minus)**2
        
        GtermPlus = exptermPlus*((Whitt2Plus*Whitt1Plus.conjugate()).real - s * abs(Whitt1Plus)**2)/r
        GtermMinus = exptermMinus*((Whitt2Minus*Whitt1Minus.conjugate()).real - s * abs(Whitt1Minus)**2)/r
    
    bdrF = np.zeros((ntr, 3))

    for i in range(ntr):
        scale = (kh/a)**(i+4)
        bdrF[i, 0] = prefac*scale*(EtermPlus + (-1)**i * EtermMinus)
        bdrF[i, 1] = prefac*scale*(BtermPlus + (-1)**i * BtermMinus)
        bdrF[i, 2] = prefac*scale*(GtermPlus - (-1)**i * GtermMinus)

    return bdrF

def EoMF(dphidt, Iterm, F, bdrF, a, H, sigma, kh):
    #F[n,0]: ErotnE
    #F[n,1]: BrotnB
    #F[n,2]: -1/2(ErotnB + BrotnE)
    #bdrF: Boundary terms
    
    ntr = F.shape[0]
    
    dFdt = np.zeros(F.shape)
    for n in range(ntr-1):
        dFdt[n,0] = (bdrF[n, 0] - ((4+n)*H + 2*a**(alpha) * sigma)*F[n,0]
                     - 2*a**(alpha)*F[n+1,2] + 2*Iterm*F[n,2]*dphidt)
        
        dFdt[n,1] = bdrF[n, 1] - ((4+n)*H)*F[n,1] + 2*a**(alpha)*F[n+1,2]
        
        dFdt[n,2] = (bdrF[n, 2] - ((4+n)*H + a**(alpha) * sigma)*F[n,2]
                     + a**(alpha)*(F[n+1,0] - F[n+1,1]) + Iterm*F[n,1]*dphidt)
    
    
    dFdt[-1,:] = EoMFtruncate(dphidt, Iterm, F[-1,:], F[-2,:], bdrF[-1,:], a, H, sigma, kh, ntr)
    
    return dFdt

def EoMFtruncate(dphidt, Iterm, F, Fmin1, bdrF, a, H, sigma, kh, ntr):
    #F[n,0]: ErotnE
    #F[n,1]: BrotnB
    #F[n,2]: -1/2(ErotnB + BrotnE)
    #bdrF: Boundary terms
    
    dFdt = np.zeros(3)
    
    dFdt[0] = bdrF[0] - ( ((4+ntr-1)*H + 2*a**(alpha) * sigma)*F[0]
                         - 2*kh**2 * a**(alpha-2)*Fmin1[2] + 2*Iterm*F[2]*dphidt)
    dFdt[1] = bdrF[1] - (4+ntr-1)*H*F[1] + 2*kh**2 * a**(alpha-2)*Fmin1[2]
    dFdt[2] = bdrF[2] - (((4+ntr-1)*H + a**(alpha) * sigma)*F[2] 
                         + kh**2 * a**(alpha-2)*(Fmin1[0] - Fmin1[1]) + Iterm*F[1]*dphidt)
    
    return dFdt

def g(x):
    if (x < 0):
        return -1
    elif (x > 0):
        return 1
    else:
        return "a miracle"
    
    
def approxPosE(xi):
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
    return t1+t2+t3+t4+t5+t6+t7+t8+t9

def approxMinE(xi):
    t1 = 1
    t2 = -9/(2**(10)*xi**2)
    t3 = 2059/(2**(21)*xi**4)
    t4 = -448157/(2**31*xi**6)
    return np.sqrt(2)*(t1 + t2 + t3 + t4)

def approxPosB(xi):
    xi = abs(xi)
    g1 = math.gamma(2/3)**2
    g2 = math.gamma(1/3)**2
    t1 = (2/3)**(1/3)*g2*xi**(1/3)/(np.pi)
    t2 = 2*np.sqrt(3)/(35*xi)
    t3 = -4*(2/3)**(1/3)*g2/(225*np.pi*xi**(5/3))
    t4 = 9*(3/2)**(1/3)*g1/(1225*np.pi*xi**(7/3))
    t5 = 132*np.sqrt(3)/(56875*xi**3)
    t6 = -9511*(2/3)**(1/3)*g2/(5457375*np.pi*xi**(11/3))
    t7 = 1448*(3/2)**(1/3)*g1/(1990625*np.pi*xi**(13/3))
    t8 = 1187163*np.sqrt(3)/(1323765625*xi**5)
    t9 = -22862986*(2/3)**(1/3)*g2/(28465171875*np.pi*xi**(17/3))
    return t1+t2+t3+t4+t5+t6+t7+t8+t9

def approxMinB(xi):
    t1 = 1
    t2 = 11/(2**(10)*xi**2)
    t3 = -2397/(2**(21)*xi**4)
    t4 = 508063/(2**31*xi**6)
    return 1/np.sqrt(2)*(t1 + t2 + t3 + t4)

def approxPosG(xi):
    xi = abs(xi)
    g1 = math.gamma(2/3)**2
    g2 = math.gamma(1/3)**2
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
    return t1+t2+t3+t4+t5+t6+t7+t8+t9+t10

def approxMinG(xi):
    t1 = 1
    t2 = -67/(2**(10)*xi**2)
    t3 = 21543/(2**(21)*xi**4)
    t4 = -6003491/(2**31*xi**6)
    return -np.sqrt(2)/(32*abs(xi))*(t1 + t2 + t3 + t4)
























