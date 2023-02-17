import numpy as np
from mpmath import whitw
from scipy.integrate import quad
import pandas as pd
import os
from EoM import *

alpha = 0

def ConstHGEF(y, t, HConst, dVdsc, dIdsc):
    #y: a 3*ntr + 2 array containing:
        #y[0]: xi
        #y[1]: lnkh
        #y[2 + 3*k]: ErotnE
        #y[2 + 3*k+1]: BrotnB
        #y[2 + 3*k+2]:1/2( ErotnB + BrotnE )
    #Hconst: Constant H value in physical time
    #dVdcs: constant potential value
    
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

    dphidt = 2*y[0]*H/dIdsc
    ddphiddt = EoMphi(dphidt, dVdsc, dIdsc, F[0,2], a, H)[1]
    dydt[0] = dIdsc*ddphiddt/(2*H) - alpha*H*y[0]
    
    kh, dydt[1], bdrF = BoundaryComputations(np.exp(y[1]), dphidt, ddphiddt, dIdsc, 0., a, H, ntr)

    dFdt = EoMF(dphidt, dIdsc, F, bdrF, a, H, 0., kh)

    dydt[2:] = dFdt.reshape(ntr*3)
    
    return dydt

def SetupConstH(x, HConst, a, ntr, Iterm):
    y = np.zeros(2+3*ntr)
    xi = float(x)
    y[0] = xi
    y[1] = np.log(2*a*HConst*xi)
    
    pwd = os.getcwd()
    filename = "ConstH_Input/ConstH_xi_" + x + "_Initialiser.dat"
    path = os.path.join(pwd, filename)
    file = os.path.exists(path)

    if(not file):
        F = np.zeros((ntr, 3))
        for i in range(ntr):
            F[i,:] = ComputeEBGn(xi, a, HConst, i)
            print(str(int(3*(i+1))) + " out of " + str(int(3*ntr)) + " bilinear terms computed")

        DataDic = dict(E = list(F[:,0]), B = list(F[:,1]), G = list(F[:,2]))
        
        output_df = pd.DataFrame(DataDic)  
        output_df.to_csv(filename)
        
            
    else:
        input_df = pd.read_table(filename, sep=",")
        data = input_df.values
        if (np.shape(data)[0]<ntr):
            nprog = np.shape(data)[0]
            print("need to compute "+str(int(3*(ntr-nprog))) + " more bilinear terms")
            F = np.zeros((ntr, 3))
            F[:nprog,0] = data[:,1].T
            F[:nprog,1] = data[:,2].T
            F[:nprog,2] = data[:,3].T
            
            for i in range(nprog, ntr):
                F[i,:] = ComputeEBGn(xi, a, HConst, i)
                print(str(int(3*(i+1-nprog))) + " out of " + str(int(3*(ntr-nprog))) + " bilinear terms computed")

            DataDic = dict(E = list(F[:,0]), B = list(F[:,1]), G = list(F[:,2]))
        
            output_df = pd.DataFrame(DataDic)  
            output_df.to_csv(filename)
        else:
            F = np.array([data[:ntr,1], data[:ntr,2], data[:ntr,3]]).T
        
    y[2:] = F.reshape(3*ntr)
    
    Vprime = ConstPotentialSlope(xi, F[0,2], HConst, Iterm)
    
    return y, Vprime

def EBGnIntegrand(k, xi, a, HConst, n):
    lam = np.array([1, -1])
    expterm = np.exp(lam*np.pi*xi)

    Whitt1Plus = whitw(-xi*(1j), 1/2, -2j*k/(a*HConst))
    Whitt2Plus = whitw(1-xi*(1j), 1/2, -2j*k/(a*HConst))
    
    Whitt1Minus = whitw(xi*(1j), 1/2, -2j*k/(a*HConst))
    Whitt2Minus = whitw(1+xi*(1j), 1/2, -2j*k/(a*HConst))
    
    EtermPlus = abs((1j*k/(a*HConst) - 1j*xi) * Whitt1Plus + Whitt2Plus)**2
    EtermMinus = abs((1j*k/(a*HConst) + 1j*xi) * Whitt1Minus + Whitt2Minus)**2
    
    BtermPlus = abs(Whitt1Plus)**2
    BtermMinus = abs(Whitt1Minus)**2
    
    GtermPlus = (Whitt2Plus*Whitt1Plus.conjugate()).real
    GtermMinus = (Whitt2Minus*Whitt1Minus.conjugate()).real
    
    #ErotnE = int(Edk)
    E = k**(1+n) * a**(-n-2) * HConst**2/(4*np.pi**2) * (expterm[0]*EtermPlus + (-1)**n * expterm[1]*EtermMinus)
    
    #BrotnB = int(Bdk)
    B = k**(3+n) * a**(-n-4) / (4*np.pi**2) * (expterm[0]*BtermPlus + (-1)**n * expterm[1]*BtermMinus)
    
    #-ErotnB = int(Gdk)
    G =  k**(2+n) * a**(-n-3) * HConst**2/(4*np.pi**2 ) * (expterm[0]*GtermPlus - (-1)**n * expterm[1]*GtermMinus)
    return E, B, G

def ComputeEBGn(xi, a, HConst, n):
    
    E = lambda k: EBGnIntegrand(k, xi, a, HConst, n)[0]
    B = lambda k: EBGnIntegrand(k, xi, a, HConst, n)[1]
    G = lambda k: EBGnIntegrand(k, xi, a, HConst, n)[2]
    
    kh = 2*a*HConst*xi
    
    En = quad(E, 0, kh)[0]
    Bn = quad(B, 0, kh)[0]
    Gn = quad(G, 0, kh)[0]
    
    return En, Bn, Gn

def ConstPotentialSlope(xi, G, HConst, Iterm):
    return -(2*(3-alpha)*HConst**2*xi/Iterm + Iterm*G)

