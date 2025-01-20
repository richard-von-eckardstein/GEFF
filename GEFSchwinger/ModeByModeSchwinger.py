import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import os

alpha=0

def ModeEoM(A, k, a, SclrCpl, sigmaE, sigmaB):
    #Compute dA2dt2 for a given gauge-field mode with wavenumber k for both helicities
    #A is the mode function and its derivative, k is the wavenumber, SclrCpl is phidot*dI2, a is the scale factor
    
    dAdt = np.zeros(A.size)
    
    drag = a**(alpha) * sigmaE
    dis1 = k * a**(alpha-1)
    dis2 = SclrCpl + a**(alpha) * sigmaB

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
    def __init__(x, G, kdep=False):
        
    #Initialise the ModeByMode class, defines all relevant quantities for this class from the background GEF values G
        x.__kdep = kdep

        if G.units: G.Unitless()
        x.__t = G.vals["t"]
        x.__N = G.vals["N"]
        kh = G.vals["kh"]
        H = G.vals["H"]
        a = G.vals["a"]
        x.__beta=G.beta

        x.__af = CubicSpline(x.__t, a)
        x.__SclrCplf = CubicSpline( x.__t, G.dIdphi()*G.vals["dphi"] )
        x.__sigmaB = G.vals["sigmaB"]
        x.__sigmaE = G.vals["sigmaE"]
        x.__delta = G.vals["delta"]
        x.__khf = CubicSpline(x.__t, kh)

        if kdep:
            x.__kFerm = G.vals["kS"]
        
        deta = lambda t, y: 1/x.__af(t)
        
        soleta = solve_ivp(deta, [min(x.__t), max(x.__t)], np.array([-1]), t_eval=x.__t)

        x.__etaf = CubicSpline(x.__t, soleta.y[0,:])

        Nend = G.EndOfInflation()[0]

        maxN = min(max(x.__N), Nend+1)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        x.maxk = CubicSpline(x.__N, kh)(maxN)
        x.mink = 10**4*kh[0]
        
        return
    
    def InitialKTN(x, init, mode="t"):
        #For a given value of time t or wavenumber k, find when k=10^(-5/2)kh(t) -> initialisation of mode equation
        #init: an array of either t: time, k: wavenumber or N: e-folds
        #mode: specify if init is time, wavenumber, or e-folds
        if mode=="t":
            tstart = init
            k = 10**(5/2)*x.__khf(tstart)
        elif mode=="N":
            tstart = CubicSpline(x.__N, x.__t)(init)
            k = 10**(5/2)*x.__khf(tstart)
        elif mode=="k":
            k = init
            dk = np.log(init[1]/init[0])
            print(dk)
            x0 = np.log(k[0]) - 5/2*np.log(10) - np.log(2)
            tstart = []
            for l in k:
                f = lambda t: np.log(l) - np.log(x.__khf(t)) - 5/2*np.log(10)
                ttmp = fsolve(f, x0, xtol=dk*1e-3)[0]
                x0 = ttmp
                tstart.append(ttmp)
            tstart = np.array(tstart)
        else:
            print("not a valid choice")
            return

        return k, tstart
    
    def ComputeMode(x, k, tstart, teval=[], atol=1e-2, rtol=1e-4):
        #For a given mode k=10^(5/2) k_h(tstart), solve the mode equation. teval: time points at which to evaluate the mode
        
        #Parse input
        tmax = max(x.__t)

        if len(teval)==0:
            teval=x.__t
        
        #For k-Dep-DampiSng: Ensure that Damping has not yet occured at the initialisation point
        if x.__kdep:
            tcross = x.__t[np.where(x.__kFerm/k < 1)][-1]
            if tstart > tcross: tstart = tcross
            sigmaEk = np.where( x.__kFerm < k, 0, x.__sigmaE )
            sigmaBk = np.where( x.__kFerm < k, 0, x.__sigmaB )

            sigmaEf = CubicSpline(x.__t, sigmaEk)
            sigmaBf = CubicSpline(x.__t, sigmaBk)

            deltaf  = lambda x: 1.0 #we always initialse modes while k > kFerm
        else:
            sigmaEf = CubicSpline(x.__t, x.__sigmaE)
            sigmaBf = CubicSpline(x.__t, x.__sigmaB)
            deltaf  = CubicSpline(x.__t, x.__delta)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        eta = x.__etaf(teval)
        teval = teval[istart:]

        #Initial conditions for A and dAdt for both helicities
        Aini = np.array([1., -1/2*sigmaEf(tstart)/k, 0, -1.,
                         1., -1/2*sigmaEf(tstart)/k, 0, -1.])*np.sqrt( deltaf(tstart) )
        
        #Define ODE to solve
        ode = lambda t, y: ModeEoM( y, k, x.__af(t), x.__SclrCplf(t), sigmaEf(t), sigmaBf(t) )
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tstart, tmax], Aini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)
        
        #the mode was in vacuum before tstart
        
        vac = list( ( np.exp(-1j*eta*k)*np.sqrt( deltaf(teval) ) )[:istart] )
        dvac = list( ( -1j*np.exp(-1j*eta*k)*np.sqrt( deltaf(teval) ) )[:istart] )

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
        #AP = sqrt(2k)*A(t,k,+), AM = sqrt(2k)*A(t,k,-), dAP = sqrt(2/k)*dAdt(t,k,+), dAM = sqrt(2/k)*dAdt(t,k,-)

        Es = []
        Bs = []
        Gs = []
        for i, k in enumerate(ks):
            if k < x.__khf(t) and k > x.mink:
                Ep, Bp, Gp = x.EBGnSpec(k, AP[i], dAP[i], 1.0, x.__af(t), n)
                Em, Bm, Gm = x.EBGnSpec(k, AM[i], dAM[i], -1.0, x.__af(t), n)
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
        N = list(np.log(x.__af(t)))
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
            filename = "Modes+Beta" + str(x.__beta) + "+M6_16" + ".dat"
        else:
            filename = name
    
        DirName = os.getcwd()
    
        path = os.path.join(DirName, filename)
    
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)
        
        return

    def ReadMode(x, file=None):
        if(file==None):
            filename = "Modes+Beta" + str(x.__beta) + "+M6_16" + ".dat"
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
    
    
    
    
    
    
    
    
    
    