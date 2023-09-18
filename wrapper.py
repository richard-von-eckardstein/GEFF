import numpy as np
from mpmath import whitw
from scipy.integrate import quad
import pandas as pd
import os
from EoM import *

alpha = 0

def fullGEF_NoSE(t, y, omega=1, f=1, approx=False):
    #y: a 3*ntr + 3 array containing:
        #y[0]: a
        #y[1]: phi
        #y[2]: dphidt
        #y[3]: lnkh
        #y[4 + 3*k]: ErotnE
        #y[4 + 3*k+1]: BrotnB
        #y[4 + 3*k+2]:1/2( ErotnB + BrotnE )
        
    #Corresponds to ntr-1 
    ratio = omega/f
    ntr = int((y.size - 4)/3)
    
    dydt = np.zeros(y.shape)
    
    #Scale Factor
    a = np.exp(y[0])
    
    #Inflaton
        #sc[0]: phi
        #sc[1]: dphidt
    sc = np.array([y[1], y[2]])
    
    #Cut Off scale:
    lnkh = y[3]
    
    #Gauge Field VeVs
        #F[n,0]: ErotnE
        #F[n,1]: BrotnB
        #F[n,2]:1/2( ErotnB + BrotnE )
    F = y[4:]
    F = F.reshape(ntr, 3)
    
    #compute H, potential, couplings and derivatives (once per timestep)
    V, dVdsc = potential(f*sc[0])/(f*omega)**2, dVdphi(f*sc[0])/(omega**2*f)
    dIdsc, ddIddsc = f*dIdphi(f*sc[0]), f**2*ddIddphi(f*sc[0])
    Hsq = FriedmannEq(a, sc[1], V, F[0,0], F[0,1], 0., ratio)
    H = np.sqrt(Hsq)
    dydt[0] = H
    
    #Scalar Field Evolution
    dscdt = EoMphi(sc[1], dVdsc, dIdsc, F[0,2], a, H, ratio)
    dydt[1] = dscdt[0]
    dydt[2] = dscdt[1]
    
    #Boundary Terms
    kh, dydt[3], bdrF = BoundaryComputations(np.exp(y[3]), dscdt[0], dscdt[1], dIdsc, ddIddsc, 
                                             a, H, ntr, 0., 0., 1.0, approx=approx)
    #Gauge-field evolution
    dFdt = EoMF(sc[1], dIdsc, F, bdrF, a, H, 0., 0., kh)
    dydt[4:] = dFdt.reshape(ntr*3)
    
    return dydt

def fullGEF_SE_collinear(t, y, omega=1, f=1, frac=0., approx=False):
    #y: a 3*ntr + 3 array containing:
        #y[0]: a
        #y[1]: phi
        #y[2]: dphidt
        #y[3]: lnkh
        #y[4]: Delta
        #y[5]: rhoChi
        #y[6 + 3*k]: ErotnE
        #y[6 + 3*k+1]: BrotnB
        #y[6 + 3*k+2]:1/2( ErotnB + BrotnE )
    #frac: smoothly vary electric and magnetic contribution
        #frac = -1: pure electric
        #frac = 1: pure magnetic
        #frac = 0: collinear mixed picture
        
    #Corresponds to ntr-1 
    ratio = omega/f
    ntr = int((y.size - 6)/3)
    
    dydt = np.zeros(y.shape)

    #Scale Factor
    a = np.exp(y[0])

    #Inflaton
        #sc[0]: phi
        #sc[1]: dphidt
    sc = np.array([y[1], y[2]])
    
    #Cut Off scale:
    lnkh = y[3]

    #Damping Factor
    delta = y[4]
    
    #Fermion Energy Density:
    rhoChi = y[5]
    
    #Gauge Field VeVs
        #F[n,0]: ErotnE
        #F[n,1]: BrotnB
        #F[n,2]:1/2( ErotnB + BrotnE )
    F = y[6:]
    F = F.reshape(ntr, 3)
    
    #compute H, potential, couplings and derivatives (once per timestep)
    V, dVdsc = potential(f*sc[0])/(f*omega)**2, dVdphi(f*sc[0])/(omega**2*f)
    dIdsc, ddIddsc = f*dIdphi(f*sc[0]), f**2*ddIddphi(f*sc[0])
    Hsq = FriedmannEq(a, sc[1], V, F[0,0], F[0,1], rhoChi, ratio)
    H = np.sqrt(Hsq)
    dydt[0] = H
    
    #Scalar Field Evolution
    dscdt = EoMphi(sc[1], dVdsc, dIdsc, F[0,2], a, H, ratio)
    dydt[1] = dscdt[0]
    dydt[2] = dscdt[1]
    
    #Conductivities
    sigmaE, sigmaB = ComputeSigmaCollinear(F[0,0], F[0,1], np.sign(F[0,2]), H*a**(-alpha), ratio, frac)
    
    #Damping Factor
    dydt[4] = EoMDelta(sigmaE, a, delta)
    
    #Fermion Energy-Density evolution
    dydt[5] = a**(alpha)*(sigmaE*F[0,0] - sigmaB*F[0,2])  - 4*H*rhoChi
    
    #Boundary Terms
    kh, dydt[3], bdrF = BoundaryComputations(np.exp(y[3]), dscdt[0], dscdt[1], dIdsc, ddIddsc, 
                                             a, H, ntr, sigmaE, sigmaB, delta, approx=approx)
    
    #Gauge-field evolution
    dFdt = EoMF(sc[1], dIdsc, F, bdrF, a, H, sigmaE, sigmaB, kh)
    dydt[6:] = dFdt.reshape(ntr*3)

    return dydt

def fullGEF_SE_mixed(t, y, omega=1, f=1, approx=False):
    #y: a 3*ntr + 3 array containing:
        #y[0]: a
        #y[1]: phi
        #y[2]: dphidt
        #y[3]: lnkh
        #y[4]: Delta
        #y[5]: rhoChi
        #y[6 + 3*k]: ErotnE
        #y[6 + 3*k+1]: BrotnB
        #y[6 + 3*k+2]:1/2( ErotnB + BrotnE )
        
    #Corresponds to ntr-1 
    ratio = omega/f
    ntr = int((y.size - 6)/3)
    
    dydt = np.zeros(y.shape)
    
    #Scale Factor
    a = np.exp(y[0])

    #Inflaton
        #sc[0]: phi
        #sc[1]: dphidt
    sc = np.array([y[1], y[2]])

    #Cut Off scale:
    lnkh = y[3]

    #Integrated Conductivity
    delta = y[4]
    
    #Fermion Energy Density:
    rhoChi = y[5]
    
    #Gauge Field VeVs
        #F[n,0]: ErotnE
        #F[n,1]: BrotnB
        #F[n,2]:1/2( ErotnB + BrotnE )
    F = y[6:]
    F = F.reshape(ntr, 3)
    
    #compute H, potential, couplings and derivatives (once per timestep)
    V, dVdsc = potential(f*sc[0])/(f*omega)**2, dVdphi(f*sc[0])/(omega**2*f)
    dIdsc, ddIddsc = f*dIdphi(f*sc[0]), f**2*ddIddphi(f*sc[0])
    Hsq = FriedmannEq(a, sc[1], V, F[0,0], F[0,1], rhoChi, ratio)
    H = np.sqrt(Hsq)
    dydt[0] = H
    
    #Scalar Field Evolution
    dscdt = EoMphi(sc[1], dVdsc, dIdsc, F[0,2], a, H, ratio)
    dydt[1] = dscdt[0]
    dydt[2] = dscdt[1]
    
    #Conductivities
    sigmaE, sigmaB = ComputeImprovedSigma(F[0,0], F[0,1], F[0,2], H*a**(-alpha), ratio)
    
    #Damping Factor
    dydt[4] = EoMDelta(sigmaE, a, delta)
    
    #Fermion Energy-Density evolution
    dydt[5] = a**(alpha)*(sigmaE*F[0,0] - sigmaB*F[0,2])  - 4*H*rhoChi
    
    #Boundary Terms
    kh, dydt[3], bdrF = BoundaryComputations(np.exp(y[3]), dscdt[0], dscdt[1], dIdsc, ddIddsc, 
                                             a, H, ntr, sigmaE, sigmaB, delta, approx=approx)
    
    #Gauge-field evolution
    dFdt = EoMF(sc[1], dIdsc, F, bdrF, a, H, sigmaE, sigmaB, kh)
    dydt[6:] = dFdt.reshape(ntr*3)
    
    return dydt
    
def ConstHGEF(y, t, HConst, dVdsc, dIdsc, omega=1, f=1, approx=False):
    #y: a 3*ntr + 2 array containing:
        #y[0]: xi
        #y[1]: lnkh
        #y[2 + 3*k]: ErotnE
        #y[2 + 3*k+1]: BrotnB
        #y[2 + 3*k+2]:1/2( ErotnB + BrotnE )
    #Hconst: Constant H value in physical time
    #dVdcs: constant potential value
    
    ratio = omega/f
    
    #Corresponds to ntr-1 
    ntr = int((y.size - 2)/3)
    
    dVdsc = dVdsc/(omega**2*f)
    dIdsc = f*dIdsc
    
    dydt = np.zeros(y.shape)
    
    #Scale Factor
    a = np.exp(HConst*t)

    #H value in alpha-time
    H = a**(alpha)*HConst
    
    #Cut Off scale:
    lnkh = y[1]
    
    #Gauge Field VeVs
        #F[n,0]: ErotnE
        #F[n,1]: BrotnB
        #F[n,2]:1/2( ErotnB + BrotnE )
    F = y[2:]
    F = F.reshape(ntr, 3)
    
    dphidt = 2*y[0]/dIdsc

    ddphiddt = EoMphi(dphidt, dVdsc, dIdsc, F[0,2], a, H, ratio)[1]
    dydt[0] = dIdsc*ddphiddt/(2*H) - alpha*H*y[0]
    
    kh, dydt[1], bdrF = BoundaryComputations(np.exp(y[1]), dphidt, ddphiddt, dIdsc, 0., a, H, ntr, approx=approx)

    dFdt = EoMF(dphidt, dIdsc, F, bdrF, a, H, 0., kh)
    
    dydt[2:] = dFdt.reshape(ntr*3)
    
    return dydt

def Inflation(y, omega=1, f=1):
    #y: a 3 array containing:
        #y[0]: lna
        #y[1]: phi
        #y[2]: dphidt
        
    ratio = omega/f
    
    dydt = np.zeros(y.shape)

    #Scale Factor
    a = np.exp(y[0])
    
    #Inflaton
        #sc[0]: phi
        #sc[1]: dphidt
    sc = np.array([y[1], y[2]])
    
    V, dVdsc = potential(f*sc[0])/(f*omega)**2, dVdphi(f*sc[0])/(omega**2*f)

    Hsq = FriedmannEq(a, sc[1], V, 0, 0, ratio)
    
    dydt[0] = np.sqrt(Hsq)
    
    H = np.sqrt(Hsq)
    
    dscdt = EoMphi(sc[1], dVdsc, 0., 0., a, H, ratio)
    dydt[1] = dscdt[0]
    dydt[2] = dscdt[1]

    return dydt