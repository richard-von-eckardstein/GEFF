import numpy as np
from mpmath import whitw
from scipy.integrate import quad
import pandas as pd
import os
from EoM import *

alpha = 0

def fullGEF(t, y, V, dVdphi, dIdphi, ddIddphi, sigma=0, omega=1, f=1, approx=False):
    #y: a 3*ntr + 3 array containing:
        #y[0]: a
        #y[1]: phi
        #y[2]: dphidt
        #y[3]: lnkh
        #y[4 + 3*k]: ErotnE
        #y[4 + 3*k+1]: BrotnB
        #y[4 + 3*k+2]:1/2( ErotnB + BrotnE )
    #V: potential of phi
    #dVdphi: derivative of potential
    #dIdphi: derivative of gauge field coupling
    #ddIddphi: 2nd deriv. of gauge field coupling
        
    ratio = omega/f
        
    #Corresponds to ntr-1 
    ntr = int((y.size - 4)/3)
    
    dydt = np.zeros(y.shape)

    #Scale Factor
    a = np.exp(y[0])
    
    #print("a", a)
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
    V, dVdsc = V(f*sc[0])/(f*omega)**2, dVdphi(f*sc[0])/(omega**2*f)
    dIdsc, ddIddsc = f*dIdphi(f*sc[0]), f**2*ddIddphi(f*sc[0])

    Hsq = FriedmannEq(a, sc[1], V, F[0,0], F[0,1], f, ratio)
    if(Hsq<0):
        print("Hsq:", Hsq)
        print("loga:", y[0])
        print("t", t)
        
    dydt[0] = np.sqrt(Hsq)
    
    H = np.sqrt(Hsq)
    
    dscdt = EoMphi(sc[1], dVdsc, dIdsc, F[0,2], a, H, ratio)
    dydt[1] = dscdt[0]
    dydt[2] = dscdt[1]
    
    dsigmadt=0
    
    kh, dydt[3], bdrF = BoundaryComputations(np.exp(y[3]), dscdt[0], dscdt[1], dIdsc, ddIddsc, 
                                             a, H, ntr, sigma, dsigmadt, approx=approx)
    
    dFdt = EoMF(sc[1], dIdsc, F, bdrF, a, H, sigma, kh)
    dydt[4:] = dFdt.reshape(ntr*3)
    
    return dydt

def ConstHGEF(y, t, HConst, dVdphi, dIdphi, omega=1, f=1, approx=False):
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
    
    #make dimless
    dVdsc = dVdphi/(omega**2*f)
    dIdsc = f*dIdphi
    
    dphidt = 2*y[0]*H/dIdsc
    ddphiddt = EoMphi(dphidt, dVdsc, dIdsc, F[0,2], a, H, ratio)[1]
    dydt[0] = dIdsc*ddphiddt/(2*H) - alpha*H*y[0]
    
    kh, dydt[1], bdrF = BoundaryComputations(np.exp(y[1]), dphidt, ddphiddt, dIdsc, 0., a, H, ntr, approx=approx)
    
    dFdt = EoMF(dphidt, dIdsc, F, bdrF, a, H, 0., kh)
    
    dydt[2:] = dFdt.reshape(ntr*3)
    
    return dydt