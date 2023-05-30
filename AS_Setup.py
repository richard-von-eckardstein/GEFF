import numpy as np
from mpmath import whitw
from scipy.integrate import quad
import pandas as pd
import os

alpha = 0

def SetupConstH(xi, beta, a, ntr, file=None):
    Fvec = np.zeros(3*ntr)
    
    Mpl = 1.0
    H = 2e-7*(beta/100)**(-1/2)*np.exp(-2.85*(xi-7))*Mpl
    
    ratio = (H/Mpl)
    
    if(file==None):
        F = np.zeros((ntr, 3))
        for i in range(ntr):
            #unitless, powers of H need to be restored
            F[i,:] = ComputeEBGn(xi, a, i)
            print(str(int(3*(i+1))) + " out of " + str(int(3*ntr)) + " bilinear terms computed")

        DataDic = dict(E = list(F[:,0]), B = list(F[:,1]), G = list(F[:,2]))
        
        output_df = pd.DataFrame(DataDic)  
        filename = "ConstH_xi" + str(xi) + "_Initialiser.dat"
        output_df.to_csv(filename)
                
    else:
        input_df = pd.read_table(file, sep=",")
        data = input_df.values
        if (np.shape(data)[0]<ntr):
            nprog = np.shape(data)[0]
            print("need to compute "+str(int(3*(ntr-nprog))) + " more bilinear terms")
            F = np.zeros((ntr, 3))
            F[:nprog,0] = data[:,1].T
            F[:nprog,1] = data[:,2].T
            F[:nprog,2] = data[:,3].T
            
            for i in range(nprog, ntr):
                F[i,:] = ComputeEBGn(xi, a, i)
                print(str(int(3*(i+1-nprog))) + " out of " + str(int(3*(ntr-nprog))) + " bilinear terms computed")

            DataDic = dict(E = list(F[:,0]), B = list(F[:,1]), G = list(F[:,2]))
        
            output_df = pd.DataFrame(DataDic)  
            output_df.to_csv(file)
        else:
            F = np.array([data[:ntr,1], data[:ntr,2], data[:ntr,3]]).T
        
    Fvec = F.reshape(3*ntr)
    
    lnkh = np.log(2*a*abs(xi))
    
    Vprime = ConstPotentialSlope(xi, F[0,2]*H**4, H, beta/Mpl)

    V0 = (H*Mpl)**2 * (3 - 2*xi**2/beta**2) - 0.5 * (F[0,0]+ F[0,1]) * H**4
    
    dIdphi = beta/Mpl
    
    f = Mpl
    omega = H
    
    return Fvec, lnkh, V0, Vprime, dIdphi, f, omega

def EBGnIntegrand(k, xi, a, n):
    lam = np.array([1, -1])
    expterm = np.exp(lam*np.pi*xi)

    Whitt1Plus = whitw(-xi*(1j), 1/2, -2j*k/a)
    Whitt2Plus = whitw(1-xi*(1j), 1/2, -2j*k/a)
    
    Whitt1Minus = whitw(xi*(1j), 1/2, -2j*k/a)
    Whitt2Minus = whitw(1+xi*(1j), 1/2, -2j*k/a)
    
    EtermPlus = abs((1j*k/a - 1j*xi) * Whitt1Plus + Whitt2Plus)**2
    EtermMinus = abs((1j*k/a + 1j*xi) * Whitt1Minus + Whitt2Minus)**2
    
    BtermPlus = abs(Whitt1Plus)**2
    BtermMinus = abs(Whitt1Minus)**2
    
    GtermPlus = (Whitt2Plus*Whitt1Plus.conjugate()).real
    GtermMinus = (Whitt2Minus*Whitt1Minus.conjugate()).real
    
    #ErotnE = int(Edk) 
    E = k**(1+n) * a**(-n-2) / (4*np.pi**2) * (expterm[0]*EtermPlus + (-1)**n * expterm[1]*EtermMinus)
    
    #BrotnB = int(Bdk) 
    B = k**(3+n) * a**(-n-4) / (4*np.pi**2) * (expterm[0]*BtermPlus + (-1)**n * expterm[1]*BtermMinus)
    
    #-ErotnB = int(Gdk)
    G =  k**(2+n) * a**(-n-3) / (4*np.pi**2 ) * (expterm[0]*GtermPlus - (-1)**n * expterm[1]*GtermMinus)
    return E, B, G

def ComputeEBGn(xi, a, n):
    
    E = lambda k: EBGnIntegrand(k, xi, a, n)[0]
    B = lambda k: EBGnIntegrand(k, xi, a, n)[1]
    G = lambda k: EBGnIntegrand(k, xi, a, n)[2]
    
    kh = 2*a*abs(xi)
    
    En = quad(E, 0, kh)[0]
    Bn = quad(B, 0, kh)[0]
    Gn = quad(G, 0, kh)[0]
    
    return En, Bn, Gn

def ConstPotentialSlope(xi, G, H, Iterm):
    return -(2*(3-alpha)*H**2*xi/Iterm + Iterm*G)