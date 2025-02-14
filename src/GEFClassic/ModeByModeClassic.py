import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import os

alpha=0

def ReadMode(file=None):
        if(file==None):
            filename = "Modes+Beta" + str(x.__beta) + "+M6_16" + ".dat"
            DirName = os.getcwd()
    
            file = os.path.join(DirName, filename)
            
        input_df = pd.read_table(file, sep=",")
        datayp = input_df.values
    
        x = np.arange(3,datayp.shape[1], 4)
        
        t = np.array(datayp[1:,1])
        N = np.array(datayp[1:,2])
        logk = np.array([(complex(datayp[0,y])).real for y in x])
        yp = np.array([[complex(datayp[i+1,y]) for i in range(len(N))] for y in x])
        dyp = np.array([[complex(datayp[i+1,y+1]) for i in range(len(N))] for y in x])
        ym = np.array([[complex(datayp[i+1,y+2]) for i in range(len(N))] for y in x])
        dym = np.array([[complex(datayp[i+1,y+3]) for i in range(len(N))] for y in x])

        k = 10**logk
        
        return t, N, k, yp, dyp, ym, dym

def ModeEoM(y, k, SclrCpl, a):
    """
    Compute the time derivative of the gauge-field mode and its derivatives for a fixed wavenumber at a given moment of time t (in Hubble units)

    Parameters
    ----------
    y : numpy.array
        contains the gauge-field mode and its derivatives for both helicities +/- (in Hubble units).
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, deta = a*dt
    k : float
        the wavenumber in Hubble units
    SclrCpl : float
        coupling induced by the axion velocity at time t, beta/M_P*dphidt (in Hubble units)
    a : float
        the scalefactor at time t

    Returns
    -------
    dydt : array
        an array of time derivatives of y

    """
    
    dydt = np.zeros(y.size)
    
    drag = 0
    dis1 = k * a**(alpha-1)
    dis2 = SclrCpl

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*(k/a)*a**(alpha)
    dydt[1] = -( drag * y[1] + (dis1  - lam * dis2) * y[0] )
    
    #Imaginary Part
    dydt[2] = y[3]*(k/a)*a**(alpha)
    dydt[3] = -( drag * y[3] + (dis1  - lam * dis2) * y[2] )
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*(k/a)*a**(alpha)
    dydt[5] = -( drag * y[5] + (dis1  - lam * dis2) * y[4] )
    
    #Imaginary Part
    dydt[6] = y[7]*(k/a)*a**(alpha)
    dydt[7] = -( drag * y[7] + (dis1  - lam * dis2) * y[6] )
    
    return dydt

class ModeByMode:
    """
    A class used to solve the gauge-field mode equation for axion inflation based on a given GEF solution. Can be used to internally verify the consistency of the GEF solution. All quantities throught are treated in Hubble units.
    
    ...
    
    Attributes
    ----------
    
    x.__t : array
        An increasing array of physical times tracking the evolution of the GEF system.
    x.__N : array
        An increasing array of e-Folds tracking the evolution of the GEF system.
    x.__beta : float
        The strength of the inflaton--gauge-field interaction, beta/M_P
    x.__af : function
        returns the scale factor, a(t), as a function of physical time. Obtained by interpolation of the GEF solution.
    x.__SclrCplf : function
        returns the coupling of the inflaton velocity to the gauge-field, beta/M_p*dphidt, as a function of physical time. Obtained by interpolation of the GEF solution.
    x.__khf : function
        returns the instability scale k_h(t) as a function of physical time. Obtained by interpolation of the GEF solution.
    x.__etaf : function
        returns the conformal time eta(t) as a function of physical time normalised to eta(0)=-1/H_0. Obtained by numerical integration and interpolation.
    x.maxk : float
        the maximal wavenumber k which can be resolved based on the dynamical range covered by the GEF solution
    x.mink : float
        the minimal wavenumber k which can be resolved based on the initial conditions of the GEF solution
    
    ...
    
    Methods
    -------
    
    InitialKTN()
        Determines the solution to k = 10^(5/2)*k_h(t). initial data can be given for the wavenumber k, the physical time coordinates t, or e-Folds N.
    ComputeMode()
        For a given wavenumber k satisfying k=10^(5/2)*k_h(t), initialises the gauge-field modes at time t in the Bunch-Davies vacuum and computes the time evolution within a given time interval, teval.
    EBGnSpec()
        Computes the spectrum of E rot^n E/a^n (=E[n]), B rot^n B/a^n (=B[n]), and -(E rot^n B)/a^n (=G[n]) at a given moment of time t and a helicity lambda gusing the gauge field spectrum A(t, k, lambda)
    ComputeEBGnMode()
        Computes the expectation values E rot^n E/a^n (=E[n]), B rot^n B/a^n (=B[n]), and -(E rot^n B)/a^n (=G[n]) at a given moment of time t given the gauge field spectrum A(t, k, +/-). Useful for comparing GEF results to ModeByMode results.
    """
    def __init__(x, G):
    #Initialise the ModeByMode class, defines all relevant quantities for this class from the background GEF values G
        if G.units: G.Unitless()
        x.__t = G.vals["t"]
        x.__N = G.vals["N"]
        kh = G.vals["kh"]
        H = G.vals["H"]
        x.__beta=G.beta

        x.__af = CubicSpline(x.__t, np.exp(x.__N))
        x.__SclrCplf = CubicSpline(x.__t, G.dIdphi()*G.vals["dphi"])
        x.__khf = CubicSpline(x.__t, kh)
        x.__Hf = CubicSpline(x.__t, H)
        
        deta = lambda t, y: 1/x.__af(t)
        
        soleta = solve_ivp(deta, [min(x.__t), max(x.__t)], np.array([-1]), t_eval=x.__t)

        x.__etaf = CubicSpline(x.__t, soleta.y[0,:])

        Nend = G.EndOfInflation()[0]

        maxN = min(max(x.__N), Nend+1)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        x.maxk = CubicSpline(x.__N, kh)(maxN)
        x.mink = 10**(3)*kh[0]
        
        return
    
    def InitialKTN(x, init, mode="t"):
        """
        Input
        -----
        init : array
           an array of physical time coordinates t, OR of e-Folds N, OR of wavenumbers k (within x.mink and x.maxk)
        mode : str
            if init contains physical time coordinates: mode="t"
            if init contains e-Folds: mode="N"
            if init contains wavenumbers: mode="k"

        Return
        ------
        k : array
            an array of wavenumbers k satisfying k=10^(5/2)k_h(tstart)
        tstart : array
            an array of physical time coordinates t satisfying k=10^(5/2)k_h(tstart)
        """

        if mode=="t":
            tstart = init
            k = 10**(5/2)*x.__khf(tstart)

        elif mode=="k":
            k = init
            x0 = np.log(k[0]) - np.log(x.__khf(0.)) - 5/2*np.log(10)
            tstart = []
            for i, l in enumerate(k):
                f = lambda t: np.log(l) - np.log(x.__khf(t)) - 5/2*np.log(10)
                ttmp = fsolve(f, x0)[0]
                #Update the initial guess based on the previous result
                if i < len(k)-1:
                    x0 = ttmp + np.log(k[i+1]/l)
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(x.__N, x.__t)(init)
            k = 10**(5/2)*x.__khf(tstart)

        else:
            print("not a valid choice")
            raise KeyError

        return k, tstart
    
    def ComputeMode(x, k, tstart, teval=[], atol=1e-3, rtol=1e-4):
        """
        Input
        -----
        k : float
           the wavenumber k for which the mode function A(t,k, +/-) is evolved.
        tstart : float
            the time coordinate satisfying k = 10^(5/2)k_h(tstart) needed to ensure that the modes initialised in the Bunch-Davies vacuum
        teval : array/list
            physical time points at which the mode function A(t,k,+/-) and its derivatives will be returned
            if teval=[], the mode functions are evaluated at x.__t
        atol : float
            the absolute precision of the numerical intergrator (1e-3 should be fine for all applications, lower will increase computational time)
        rtol : float
            the relative precision of the numerical integrator (1e-4 or lower for good accuracy)

        Return
        ------
        yp : array
            the positive helicity modes (rescaled), sqrt(2k)*A(teval, k, +)
        dyp : array
            the derivative of the positive helicity modes (rescaled), sqrt(2/k)*dAdeta(teval, k, +)
        ym : array
            the negative helicity modes (rescaled), sqrt(2k)*A(teval, k, -)
        dym : array
            the derivative of the negative helicity modes (rescaled), sqrt(2/k)*dAdeta(teval, k, -)
        """

        
        #Initial conditions for y and dydt for both helicities (rescaled appropriately)
        yini = np.array([1., 0, 0, -1., 1, 0, 0, -1.])
        
        #Define ODE to solve
        ode = lambda t, y: ModeEoM(y, k, x.__SclrCplf(t), x.__af(t))
        
        #maximal time
        tmax = max(x.__t)

        #parse teval input
        if len(teval)==0:
            teval=x.__t
        
        #conformal time needed for relative phases
        eta = x.__etaf(teval)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        teval = teval[istart:]
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tstart, tmax], yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)
        
        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create array of mode evolution stringing together vacuum and non-vacuum time evolutions to get evolution from t0 to tend
        yp = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )
        dyp = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )
        
        ym = np.array( vac + list( (sol.y[4,:] + 1j*sol.y[6,:])*np.exp(-1j*k*eta[istart]) ) )
        dym = np.array( dvac + list( (sol.y[5,:] + 1j*sol.y[7,:])*np.exp(-1j*k*eta[istart]) ) )

        return yp, dyp, ym, dym
    
    def EBGnSpec(x, k, lam, ylam, dylam, a, n):
        """
        Input
        -----
        k : float
            the wavenumber for which the spectrum should be computed
        lam : float
            the helicity of the spectrum (either +1 or -1)
        ylam : float
            the mode function sqrt(2k) A(t,k,lam) for a given wavenumber k and helicity lam evaluated at time t
        dylam : float
            the mode-function derivative, sqrt(2/k) dAdeta(t,k,lam) for a given wavenumber k and helicity lam evaluated at time t
        a : float
            the scale factor at time t
        n : int
            the power of the curl in E rot^n E, B rot^n B, etc.

        Return
        ------
        E : float
            the spectrum of 1/a^n E rot^n E for wavenumber k and helicity lam
        B : float
            the spectrum of 1/a^n B rot^n B for wavenumber k and helicity lam
        G : float
            the spectrum of -1/a^n E rot^n B for wavenumber k and helicity lam
        """
    
        Eterm = abs(dylam)**2

        Bterm = abs(ylam)**2

        Gterm = lam*(ylam.conjugate() * dylam).real

        #prefac modified to account for sqrt(2k) factor in modes
        prefac = lam**n * 1/(2*np.pi)**2 * (k/a)**(n+3)/a

        #ErotnE = int(Edk) 
        E = prefac * Eterm

        #BrotnB = int(Bdk) 
        B = prefac * Bterm

        #-ErotnB = int(Gdk)
        G = prefac * Gterm
        return E, B, G
    
    def ComputeEBGnMode(x, yP, yM, dyP, dyM, t, ks, n=0):
        """
        Input
        -----
        yP : array
            the positive-helicity mode sqrt(2ks)*A(t,ks,+) for a fixed time t
        yM : array
            the negative-helicity mode sqrt(2ks)*A(t,ks,-) for a fixed time t
        dyP : array
            the positive-helicity mode's derivative sqrt(2/ks)*dAdeta(t,ks,+) for a fixed time t
        dyM : array
            the negative-helicity mode's derivative sqrt(2/ks)*dAdeta(t,ks,+) for a fixed time t
        t : float
            the physical time at which to evaluate the function
        ks : array
            an array of wavenumbers associated to the modes
        n : int
            the power of the curl in E rot^n E, B rot^n B, etc.

        Return
        ------
        En : float
            the value of 1/a^n E rot^n E at time t
        Bn : float
            the value of 1/a^n B rot^n B at time t
        Gn : float
            the value of -1/a^n E rot^n B at time t
        """
        Es = []
        Bs = []
        Gs = []
        for i, k in enumerate(ks):
            if k < x.__khf(t) and k > x.mink:
                Ep, Bp, Gp = x.EBGnSpec(k, 1.0, yP[i], dyP[i], x.__af(t), n)
                Em, Bm, Gm = x.EBGnSpec(k, -1.0, yM[i], dyM[i], x.__af(t), n)
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

    def SaveMode(x, t, ks, yp, dyp, ym, dym, name=None):
        logk = np.log10(ks)
        N = list(np.log(x.__af(t)))
        N = np.array([np.nan]+N)
        t = np.array([np.nan]+list(t))
        dic = {"t":t}
        dic = dict(dic, **{"N":N})
        for k in range(len(logk)):
            dictmp = {"yp_" + str(k) :np.array([logk[k]] + list(yp[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"ym_" + str(k) :np.array([logk[k]] + list(dyp[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"dyp_" + str(k):np.array([logk[k]] + list(ym[k,:]))}
            dic = dict(dic, **dictmp)
            dictmp = {"dym_" + str(k):np.array([logk[k]] + list(dym[k,:]))}
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

    