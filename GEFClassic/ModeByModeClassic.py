import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
import pandas as pd
import math
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import os
from importlib import reload
import sys
sys.path.append(os.getcwd()+"/../../GEF")
import GEFClassic
from GEFClassic import GEF

alpha=0

def ModeEoM(A, k, SclrCpl, a):
    #Compute dA2dt2 for a given gauge-field mode with wavenumber k for both helicities
    #A is the mode function and its derivative, k is the wavenumber, SclrCpl is phidot*dI2, a is the scale factor
    
    dAdt = np.zeros(A.size)
    
    drag = 0
    dis1 = k * a**(alpha-1)
    dis2 = SclrCpl

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

class ModeByMode:
    #Class to compute the gauge-field mode time evolution and the E2, B2, EB quantum expectation values from the modes
    def __init__(x, G):
        
    #Initialise the ModeByMode class, defines all relevant quantities for this class from the background GEF values G
        if G.units: G.Unitless()
        x.t = G.vals["t"]
        x.N = G.vals["N"]
        kh = G.vals["kh"]
        H = G.vals["H"]
        x.beta=G.beta

        x.af = CubicSpline(x.t, np.exp(x.N))
        x.SclrCplf = CubicSpline(x.t, G.dIdphi()*G.vals["dphi"])
        x.khf = CubicSpline(x.t, kh)
        
        deta = lambda t, y: 1/x.af(t)
        
        soleta = solve_ivp(deta, [min(x.t), max(x.t)], np.array([-1]), t_eval=x.t)

        x.etaf = CubicSpline(x.t, soleta.y[0,:])

        Nend = G.EndOfInflation()[0]

        maxN = min(max(x.N), Nend+1)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        x.maxk = CubicSpline(x.N, kh)(maxN)
        x.mink = 10**(3)*kh[0]
        
        return
    
    def InitialKTN(x, init, mode="t"):
        #For a given value of time t or wavenumber k, find when k=10^(-5/2)kh(t) -> initialisation of mode equation
        #init: an array of either t: time, k: wavenumber or N: e-folds
        #mode: specify if init is time, wavenumber, or e-folds
        if mode=="t":
            tstart = init
            k = 10**(5/2)*x.khf(tstart)
        elif mode=="k":
            k = init
            x0 = np.log(k[0]) - 5/2*np.log(10)
            tstart = []
            for l in k:
                f = lambda t: np.log(l) - np.log(x.khf(t)) - 5/2*np.log(10)
                ttmp = fsolve(f, x0)[0]
                x0 = ttmp
                tstart.append(ttmp)
            tstart = np.array(tstart)
        elif mode=="N":
            tstart = CubicSpline(x.N, x.t)(init)
            k = 10**(5/2)*x.khf(tstart)
        else:
            print("not a valid choice")
            return

        return k, tstart
    
    def ComputeMode(x, k, tstart, teval=[]):
        #For a given mode k=10^(5/2) k_h(tstart), solve the mode equation. teval: time points at which to evaluate the mode
        
        #Initial conditions for A and dAdt for both helicities (rescaled appropriately)
        Aini = np.array([1., 0, 0, -1., 1, 0, 0, -1.])
        
        #Define ODE to solve
        ode = lambda t, y: ModeEoM(y, k, x.SclrCplf(t), x.af(t))
        
        tmax = max(x.t)

        if len(teval)==0:
            teval=x.t
            
        eta = x.etaf(teval)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        teval = teval[istart:]
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tstart, tmax], Aini, t_eval=teval, method="RK45", atol=1e-8, rtol=1e-8)
        
        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create array of mode evolution stringing together vacuum and non-vacuum time evolutions to get evolution from t0 to tend
        Ap = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )
        dApdt = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )
        
        Am = np.array( vac + list( (sol.y[4,:] + 1j*sol.y[6,:])*np.exp(-1j*k*eta[istart]) ) )
        dAmdt = np.array( dvac + list( (sol.y[5,:] + 1j*sol.y[7,:])*np.exp(-1j*k*eta[istart]) ) )

        return Ap, dApdt, Am, dAmdt
    
    def EBGnSpec(x, k, A1, A2, lam, a, n):
        #The integrand of ErotnE, BrotnB and ErotnB for a given wavenumber k computed with A1: A(t,k) and A2: dAdt(t,k) (rescaled)
    
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
    
    def ComputeEBGnMode(x, AP, AM, dAP, dAM, t, ks, n=0):
        #Compute quantum expecation values ErotnE, BrotnB and ErotnB for a given time t and with 
        #AP = A(t,k,+), AM = A(t,k,-), dAP = dAdt(t,k,+), dAM = dAdt(t,k,-) rescaled in an appropriate way
        
        #AP AM dAP, dAM are sqrt(2k)Ap etc.
        Es = []
        Bs = []
        Gs = []
        for i, k in enumerate(ks):
            if k < x.khf(t) and k > x.mink:
                Ep, Bp, Gp = x.EBGnSpec(k, AP[i], dAP[i], 1.0, x.af(t), n)
                Em, Bm, Gm = x.EBGnSpec(k, AM[i], dAM[i], -1.0, x.af(t), n)
                Es.append(Ep + Em)
                Bs.append(Bp + Bm)
                Gs.append(Gp + Gm)
            else:
                Es.append(0)
                Bs.append(0)
                Gs.append(0)

        Es = np.array(Es)
        En = trapezoid(Es, ks)
        Bs = np.array(Bs)
        Bn = trapezoid(Bs, ks)
        Gs = np.array(Gs)
        Gn = trapezoid(Gs, ks)

        return En, Bn, Gn

    def SaveMode(x, t, ks, Ap, dAp, Am, dAm, name=None):
        logk = np.log10(ks)
        N = list(np.log(x.af(t)))
        N = np.array([np.nan]+N)
        t = np.array([np.nan]+list(t))
        dic = {"t":t}
        dic = dict(dic, **{"N":N})
        for k in range(len(logk)):
            dictmp = {"Ap_" + str(k) :np.array([logk[k]] + list(Ap[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"Am_" + str(k) :np.array([logk[k]] + list(dAp[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"dAp_" + str(k):np.array([logk[k]] + list(Am[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"dAm_" + str(k):np.array([logk[k]] + list(dAm[k,:]))}
            dic = dict(dic, **dictmp)
            
        if(name==None):
            filename = "Modes+Beta" + str(x.beta) + "+M6_16" + ".dat"
        else:
            filename = name
    
        DirName = os.getcwd()
    
        path = os.path.join(DirName, filename)
    
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)
        
        return

    def ReadMode(x, file=None):
        if(file==None):
            filename = "Modes+Beta" + str(x.beta) + "+M6_16" + ".dat"
            DirName = os.getcwd()
    
            file = os.path.join(DirName, filename)
            
        input_df = pd.read_table(file, sep=",")
        dataAp = input_df.values
    
        x = np.arange(3,dataAp.shape[1], 4)
        
        t = np.array(dataAp[1:,1])
        N = np.array(dataAp[1:,2])
        logk = np.array([(complex(dataAp[0,y])).real for y in x])
        Ap = np.array([[complex(dataAp[i+1,y]) for i in range(len(N))] for y in x])
        dAp = np.array([[complex(dataAp[i+1,y+1]) for i in range(len(N))] for y in x])
        Am = np.array([[complex(dataAp[i+1,y+2]) for i in range(len(N))] for y in x])
        dAm = np.array([[complex(dataAp[i+1,y+3]) for i in range(len(N))] for y in x])

        k = 10**logk
        
        return t, N, k, Ap, dAp, Am, dAm
    
    
    
    
    
    
    
    
    
    