import numpy as np
import matplotlib.pyplot as plt
from mpmath import whitw
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import quad
import pandas as pd
import math
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from ..Common.timer import Timer
import os

alpha = 0
Mpl = 1.

def ModeEoM(A, k, dphidt, dIdphi, a, sigmaE=0, sigmaB=0):
    #omega=1.
    dAdt = np.zeros(A.size)
    
    drag = a**(-alpha) * sigmaE
    dis1 = k * a**(alpha-1)
    dis2 = dIdphi * dphidt + a**(alpha)*sigmaB

    #positive helicity
    lam = 1.
    #Real Part
    dAdt[0] = A[1]*(k/a)*a**(alpha)
    dAdt[1] = -( drag * A[1] + (dis1  - lam * dis2) * A[0] )
    
    #Imaginary Part
    dAdt[2] = A[3]*(k/a)*a**(alpha)
    dAdt[3] = -( drag * A[3] + (dis1  - lam * dis2) * A[2] )
    
    
    #negative helicity
    lam = -1.
    #Real Part
    dAdt[4] = A[5]*(k/a)*a**(alpha)
    dAdt[5] = -( drag * A[5] + (dis1  - lam * dis2) * A[4] )
    
    #Imaginary Part
    dAdt[6] = A[7]*(k/a)*a**(alpha)
    dAdt[7] = -( drag * A[7] + (dis1  - lam * dis2) * A[6] )
    
    return dAdt

def ModeByMode(tmin, tmax, tend, kh, dphidt, dIdphi, a, steps=100, cut=0.1, sigmaE=None, sigmaB=None, delta=1.):
    dt = (tmax-tmin)/(steps)
    ts = np.arange(tmin, tend, dt)
    
    #Convert to Function in case no SE
    if sigmaE==None:
        sigmaE = lambda x: 0.
        delta = lambda x: 1.
    if sigmaB==None:
        sigmaB = lambda x: 0.
    
    Ap = []
    dApdt = []
    Am = []
    dAmdt = []
    ks = []
    t2 = Timer()
    t2.start()
    
    for i in range(1,steps):
        s = ts[i]
        if (kh(s)/kh(ts[i-1]) > cut): 
            k = 10**(5/2)*kh(s)
            ks.append(k)
        
            #Initialise Modes
            Aini = np.array([1., 0, -1/2*sigmaE(s)/k, -1., 1, 0, -1/2*sigmaE(s)/k, -1.])*np.sqrt(delta(s))
            #Define ModeEquation
            ode = lambda x, y: ModeEoM(y, k, dphidt(x), dIdphi, a(x), sigmaE=sigmaE(x), sigmaB=sigmaB(x))
            
            sol = solve_ivp(ode, [s, ts[-1]], Aini, t_eval=ts[i:], method="RK45", atol=1e-8, rtol=1e-8)
            
            #vacuum will not contribute to integrals, so we set this part to 0 to have equal length arrays
            vac = list((1+0j)*np.ones((i)))
            dvacdt = list((0.-1j)*np.ones((i)))
            
            Aptmp = list(np.array(sol.y[0,:]) + np.array(sol.y[2,:])*1j)
            Ap.append([*vac, *Aptmp])
            
            dAptmp = list(np.array(sol.y[1,:]) + np.array(sol.y[3,:])*1j)
            dApdt.append([*dvacdt, *dAptmp])
            
            
            Amtmp = list(np.array(sol.y[4,:]) + np.array(sol.y[6,:])*1j)
            Am.append([*vac, *Amtmp])
            
            dAmtmp = list(np.array(sol.y[5,:]) + np.array(sol.y[7,:])*1j)
            dAmdt.append([*dvacdt, *dAmtmp])
        else:
            print("skip")
            continue
            
    Ap = np.array(Ap)
    dApdt = np.array(dApdt)            
    Am = np.array(Am)
    dAmdt = np.array(dAmdt)
    
    t2.stop()    
    
    return Ap, dApdt, Am, dAmdt, ks, ts

def EBGnIntegrandMode(k, A1, A2, lam, a, n):
    
    Eterm = abs(A2)**2
    Bterm = abs(A1)**2   
    Gterm = lam*(A1.conjugate() * A2).real
    
    #prefac modified to account for sqrt(2k) factor in modes
    prefac = lam**n * 1/(2*np.pi)**2 * (k/a)**(n+3)/a
    
    #ErotnE = int(Edk) 
    E = prefac * Eterm
    
    #BrotnB = int(Bdk) 
    B = prefac * Bterm

    #-ErotnB = int(Gdk)
    G = prefac * Gterm
    return E, B, G

def ComputeEBGnMode(AP, AM, dAP, dAM, a, ks, kh, n):
    #AP AM dAP, dAM are sqrt(2k)Ap etc.
    m = len(ks)
    Es = []
    Bs = []
    Gs = []
    for k in range(m):
        Ep, Bp, Gp = EBGnIntegrandMode(ks[k], AP[k], dAP[k], 1.0, a, n)
        Em, Bm, Gm = EBGnIntegrandMode(ks[k], AM[k], dAM[k], -1.0, a, n)
        Es.append(Ep + Em)
        Bs.append(Bp + Bm)
        Gs.append(Gp + Gm)
    
    En, Bn, Gn = 0, 0, 0
    
    for k in range(1,m):
        if (ks[k-1]<kh):
            #print(kh)
            dk = ks[k]-ks[k-1]
            En += dk*(Es[k] + Es[k-1])/2
            Bn += dk*(Bs[k] + Bs[k-1])/2
            Gn += dk*(Gs[k] + Gs[k-1])/2
        else:
            return En, Bn, Gn

    return En, Bn, Gn

def ComputeEBGnModeInterp(AP, AM, dAP, dAM, a, ks, kh, n):
    m = len(ks)
    Es = []
    Bs = []
    Gs = []
    for k in range(m):
        Ep, Bp, Gp = EBGnIntegrandMode(ks[k], AP[k], dAP[k], 1.0, a, n)
        Em, Bm, Gm = EBGnIntegrandMode(ks[k], AM[k], dAM[k], -1.0, a, n)
        Es.append(Ep + Em)
        Bs.append(Bp + Bm)
        Gs.append(Gp + Gm)
        
    Es = np.array(Es)
    Bs = np.array(Bs)
    Gs = np.array(Gs)
        
    E = lambda x: CubicSpline(ks, Es)(x)
    B = lambda x: CubicSpline(ks, Bs)(x)
    G = lambda x: CubicSpline(ks, Gs)(x)
    
    kmin = min(ks)
    
    En = quad(E, kmin, kh)[0]
    Bn = quad(B, kmin, kh)[0]
    Gn = quad(G, kmin, kh)[0]

    return En, Bn, Gn

def RunMBM(file, Nstart, beta, EarlyModes=200, LateModes=800, cut=0.01, SE=False, save=None):
    input_df = pd.read_table(file, sep=",")
    data = input_df.values
    tR = data[1:,1]
    NR = data[1:,2]
    aR = data[1:,3]
    khR = data[1:,5]
    dphidtR = data[1:,7]
    ER = data[1:,10]
    BR = data[1:,11]
    GR = data[1:,12]
    af = CubicSpline(tR, aR)
    dphidtf = CubicSpline(tR, dphidtR)
    khf = CubicSpline(tR, khR)
    if(SE):
        deltaR = data[1:,13]
        sigmaER = data[1:,15]
        sigmaBR = data[1:,16]
        sigmaEf = CubicSpline(tR, sigmaER)
        sigmaBf = CubicSpline(tR, sigmaBR)
        deltaf = CubicSpline(tR, deltaR)
    else:
        sigmaEf = lambda t: 0.
        sigmaBf = lambda t: 0.
        deltaf = lambda t: 1.
        
    maxkh = max(khR)*10**(-3/2)
    wait = True
    start = False
    stop = False
    i = 0
    while(not(start and stop)):
        N = NR[i]
        kh = khR[i+1]
        if (Nstart>N):
            tini = tR[i]
        else:
            start = True
        if (kh<maxkh):
            tend = tR[i+1]
        else:
            stop = True
        i += 1
    Ap1, dAp1, Am1, dAm1, ks1, ts1 = ModeByMode(1., tini, tR[-1], khf, dphidtf, beta/Mpl, af, steps=EarlyModes,
                                                    cut=cut, sigmaE=sigmaEf, sigmaB=sigmaBf, delta=deltaf)
    Ap2, dAp2, Am2, dAm2, ks2, ts = ModeByMode(tini, tend, tR[-1], khf, dphidtf, beta/Mpl, af, steps=LateModes,
                                                    cut=cut, sigmaE=sigmaEf, sigmaB=sigmaBf, delta=deltaf)
    
    #Only Mode-functions after Nstart are returned and stored
    Ap = np.array([*[CubicSpline(ts1, Ap1[i,:])(ts) for i in range(len(ks))], *list(Ap2)])
    dAp = np.array([*[CubicSpline(ts1, dAp1[i,:])(ts) for i in range(len(ks))], *list(dAp2)])
    Am = np.array([*[CubicSpline(ts1, Am1[i,:])(ts) for i in range(len(ks))], *list(Am2)])
    dAm = np.array([*[CubicSpline(ts1, dAm1[i,:])(ts) for i in range(len(ks))], *list(dAm2)])
    ks = np.array([*list(ks1), *list(logks2)])
    
    if(save != None):
        SaveMode(ts, ks, Ap, dAp, Am, dAm, af, name=save+".dat")
    
    return ts, ks, Ap, dAp, Am, dAm, af(ts), khf(ts)