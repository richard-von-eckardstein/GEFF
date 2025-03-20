import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import os
from numpy.typing import ArrayLike

alpha=0

def ReadMode(file):        
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

def ModeEoM(y : ArrayLike, k : float, a : float, SclrCpl : float, sigmaE : float=0., sigmaB : float=0.):
    """
    Compute the time derivative of the gauge-field mode and its derivatives for a fixed comoving wavenumber at a given moment of time t

    Parameters
    ----------
    y : numpy.array
        contains the gauge-field mode and its derivatives for both helicities +/- (in Hubble units).
        y[0/4] = Re( sqrt(2k)*A(t,k,+/-) ), y[2/6] = Im( sqrt(2k)*A(t,k,+/-) )
        y[1/5] = Re( sqrt(2/k)*dAdeta(t,k,+/-) ), y[3/7] = Im( sqrt(2/k)*dAdeta(t,k,+/-) ), eta being conformal time, a*deta = dt
    k : float
        the comoving wavenumber in Hubble units
    a : float
        the scalefactor at time t
    SclrCpl : float
        coupling induced by the axion velocity at time t, beta/M_P*dphidt (in Hubble units)
    sigmaE : float
        electric damping term induced by fermions (only relevant for Schwinger effect runs)
    sigmaB : float
        magnetic damping term induced by fermions (only relevant for Schwinger effect runs)

    Returns
    -------
    dydt : array
        an array of time derivatives of y

    """
    
    dydt = np.zeros(y.size)
    
    drag = a**(alpha) * sigmaE
    dis1 = k * a**(alpha-1)
    dis2 = SclrCpl + a**(alpha) * sigmaB

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
    #Class to compute the gauge-field mode time evolution and the E2, B2, EB quantum expectation values from the modes
    def __init__(x, G):
        """
        A class used to solve the gauge-field mode equation for axion inflation based on a given GEF solution.
        Can be used to internally verify the consistency of the GEF solution. All quantities throught are treated in Hubble units.

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
            returns the coupling of the inflaton velocity to the gauge-field, beta/M_p*dphidt, as a function of physical time.
            Obtained by interpolation of the GEF solution.
        x.__khf : function
            returns the instability scale k_h(t) as a function of physical time. Obtained by interpolation of the GEF solution.
        x.__etaf : function
            returns the conformal time eta(t) as a function of physical time normalised to eta(0)=-1/H_0.
            Obtained by numerical integration and interpolation.
        x.__SE : string | None:
            if the GEF incorporates the Schwinger effect, x.__SE="KDep" or x.__SE="Old", depending on the configuartion of the GEF run (G.SEModel)
            otherwise, x.__SE=None
        x.__sigmaE : array:
            an array containing the electric conductivities as a function of time (only relevant if x.__SE != None)
        x.__sigmaB : array:
            an array containing the magnetic conductivities as a function of time (only relevant if x.__SE != None)
        x.__delta : array:
            an array containing the time accumulated damping due to electric conductivity, exp(-int[sigmaE,dt] ) as a function of time (only relevant if x.__SE == "Old") 
        x.__kFerm : array
            an array containing the fermion pair-creation scale as a function of time (only relevant if x.__SE == "KDep") 
        x.maxk : float
            the maximal comoving wavenumber k which can be resolved based on the dynamical range covered by the GEF solution
        x.mink : float
            the minimal comoving wavenumber k which can be resolved based on the initial conditions of the GEF solution

        ...

        Methods
        -------

        InitialKTN()
            Determines the solution to k = 10^(5/2)*k_h(t). 
            Initial data can be given for the comoving wavenumber k, the physical time coordinates t, or e-Folds N.
        ComputeMode()
            For a given comoving wavenumber k satisfying k=10^(5/2)*k_h(t), initialises the gauge-field modes at time t in the Bunch-Davies vacuum
            and computes the time evolution within a given time interval, teval.
        EBGnSpec()
            Computes the spectrum of E rot^n E/a^n (=E[n]), B rot^n B/a^n (=B[n]), and -(E rot^n B)/a^n (=G[n])
            at a given moment of time t and a helicity lambda gusing the gauge field spectrum A(t, k, lambda)
        ComputeEBGnMode()
            Computes the expectation values E rot^n E/a^n (=E[n]), B rot^n B/a^n (=B[n]), and -(E rot^n B)/a^n (=G[n]) at a given moment of time t
            given the gauge field spectrum A(t, k, +/-). Useful for comparing GEF results to mode-by-mode results.
        """
        
        #Initialise the ModeByMode class, defines all relevant quantities for this class from the background GEF values G
        if G.units: G.Unitless()
        x.__t = G.vals["t"]
        x.__N = G.vals["N"]
        kh = G.vals["kh"]
        x.__beta = G.beta

        x.__af = CubicSpline(x.__t, G.vals["a"])
        x.__SclrCplf = CubicSpline( x.__t, G.dIdphi()*G.vals["dphi"] )
        x.__khf = CubicSpline(x.__t, kh)

        #Assess if the GEF run incorporates Fermions
        if hasattr(G, "SEModel"):
            x.__SE = G.SEModel
            x.__sigmaB = G.vals["sigmaB"]
            x.__sigmaE = G.vals["sigmaE"]
            x.__delta = G.vals["delta"]
            
            if x.__SE=="KDep":
                x.__kFerm = G.vals["kS"]
            elif x.__SE=="Del1":
                x.__kFerm = G.vals["kh"]

        else:
            x.__SE = None
        
        deta = lambda t, y: 1/x.__af(t)
        
        soleta = solve_ivp(deta, [min(x.__t), max(x.__t)], np.array([-1]), t_eval=x.__t)

        x.__etaf = CubicSpline(x.__t, soleta.y[0,:])

        Nend = G.EndOfInflation()

        maxN = min(max(x.__N), Nend)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        x.maxk = CubicSpline(x.__N, kh)(maxN)
        x.mink = 10**4*kh[0]
        
        return
    
    def InitialKTN(x, init, mode="t"):
        """
        Input
        -----
        init : array
           an array of physical time coordinates t, OR of e-Folds N, OR of comoving wavenumbers k (within x.mink and x.maxk)
        mode : str
            if init contains physical time coordinates: mode="t"
            if init contains e-Folds: mode="N"
            if init contains comoving wavenumbers: mode="k"

        Return
        ------
        k : array
            an array of comoving wavenumbers k satisfying k=10^(5/2)k_h(tstart)
        tstart : array
            an array of physical time coordinates t satisfying k=10^(5/2)k_h(tstart)
        """

        if mode=="t":
            tstart = init
            k = 10**(5/2)*x.__khf(tstart)

        elif mode=="k":
            k = init
            t0 = 3/2*np.log(10)
            x0 = np.log(k[0]) - np.log(x.__khf(t0)) - 5/2*np.log(10)
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
    
    def ComputeMode(x, k, tstart, teval=[], atol=1e-3, rtol=1e-5):
        """
        Input
        -----
        k : float
           the comoving wavenumber k for which the mode function A(t,k, +/-) is evolved.
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

        #Setup initial modes and ODE depending on Schwinger effect mode
        if x.__SE == None:
            #Initial conditions for y and dydt for both helicities (rescaled appropriately)
            yini = np.array([1., 0, 0, -1., 1, 0, 0, -1.])
            deltaf  = lambda x: 1.0
            
            #Define ODE to solve (sigmaE=0, sigmaB=0)
            ode = lambda t, y: ModeEoM(y, k, x.__af(t), x.__SclrCplf(t))
        else:
            #Treat sigma's depending on KDep or not
            if x.__SE in ["KDep", "Del1"]:
                tcross = x.__t[np.where(x.__kFerm/k < 1)][-1]
                if tstart > tcross: tstart = tcross
                sigmaEk = np.heaviside( x.__kFerm - k, 0.5)*x.__sigmaE
                sigmaBk = np.heaviside( x.__kFerm - k, 0.5)*x.__sigmaB

                sigmaEf = CubicSpline(x.__t, sigmaEk)
                sigmaBf = CubicSpline(x.__t, sigmaBk)

                deltaf  = np.vectorize(lambda x: 1.0) #we always initialse modes while k > kFerm
            elif x.__SE=="Old":
                sigmaEf = CubicSpline(x.__t, x.__sigmaE)
                sigmaBf = CubicSpline(x.__t, x.__sigmaB)
                deltaf  = CubicSpline(x.__t, x.__delta)
                

            #Initial conditions for y and dydt for both helicities (rescaled appropriately)
            yini = np.array([1., -1/2*sigmaEf(tstart)*x.__af(tstart)/k, 0, -1.,
                             1., -1/2*sigmaEf(tstart)*x.__af(tstart)/k, 0, -1.])*np.sqrt( deltaf(tstart) )
            
            #Define ODE to solve
            ode = lambda t, y: ModeEoM( y, k, x.__af(t), x.__SclrCplf(t), sigmaEf(t), sigmaBf(t) )
        
        #parse teval input
        if len(teval)==0:
            teval=x.__t
        tmax = max(teval)
        
        #conformal time needed for relative phases
        eta = x.__etaf(teval)
        delta = deltaf(teval)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        teval = teval[istart:]
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tstart, tmax], yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)
        
        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k)*np.sqrt( delta[:istart] ) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k)*np.sqrt( delta[:istart] ) )

        #Create array of mode evolution stringing together vacuum and non-vacuum time evolutions to get evolution from t0 to tend
        yp = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )
        dyp = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )
        
        ym = np.array( vac + list( (sol.y[4,:] + 1j*sol.y[6,:])*np.exp(-1j*k*eta[istart]) ) )
        dym = np.array( dvac + list( (sol.y[5,:] + 1j*sol.y[7,:])*np.exp(-1j*k*eta[istart]) ) )

        return yp, dyp, ym, dym
    
    def EBGnSpec(x, k : float, t : float, lam : float, ylam : float, dylam : float, n : int):
        """
        Input
        -----
        k : float
            the comoving wavenumber for which the spectrum should be computed
        t : float
            the time at which to evaluate the spectrum
        lam : float
            the helicity of the spectrum (either +1 or -1)
        ylam : float
            the mode function sqrt(2k) A(t,k,lam) for a given comoving wavenumber k and helicity lam evaluated at time t
        dylam : float
            the mode-function derivative, sqrt(2/k) dAdeta(t,k,lam) for a given comoving wavenumber k and helicity lam evaluated at time t
        n : int
            the power of the curl in E rot^n E, B rot^n B, etc.

        Return
        ------
        E : float
            the spectrum of 1/a^n E rot^n E for comoving wavenumber k and helicity lam
        B : float
            the spectrum of 1/a^n B rot^n B for comoving wavenumber k and helicity lam
        G : float
            the spectrum of -1/a^n E rot^n B for comoving wavenumber k and helicity lam
        """

        a = x.__af(t)
    
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
    
    def ComputeEBGnMode(x, yP : ArrayLike, yM : ArrayLike, dyP : ArrayLike, dyM : ArrayLike, t : float, ks : ArrayLike, n : int=0):
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
            an array of comoving wavenumbers associated to the modes
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
                Ep, Bp, Gp = x.EBGnSpec(k, t,  1.0, yP[i], dyP[i], n)
                Em, Bm, Gm = x.EBGnSpec(k, t, -1.0, yM[i], dyM[i], n)
                Es.append(Ep + Em)
                Bs.append(Bp + Bm)
                Gs.append(Gp + Gm)
            else:
                Es.append(0)
                Bs.append(0)
                Gs.append(0)

        En = trapezoid(np.array(Es), ks)
        Bn = trapezoid(np.array(Bs), ks)
        Gn = trapezoid(np.array(Gs), ks)

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


    
    
    
    
    
    
    
    
    