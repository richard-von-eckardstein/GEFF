import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapezoid

from src.Tools.ModeByMode import ReadMode

from ptarcade.models_utils import g_rho, g_rho_0, g_s, g_s_0, T_0, M_pl, gev_to_hz, omega_r, h

from numpy.typing import ArrayLike

def TensorModeEoM(y : ArrayLike, k : float, H : float, a : float):
    """
    Compute the time derivative of a vacuum tensor mode and its derivative for a fixed comoving wavenumber at a given moment of time t (in Hubble units)

    Parameters
    ----------
    y : array
        contains the vacuum tensor mode and its derivative. (in Hubble units).
        y[0] = Re( sqrt(2k)*a(t)*h(t,k)/2 ), y[2] = Im( sqrt(2k)*a(t)*h(t,k)/2  )
        y[1/5] = Re( sqrt(2/k)*a(t)*dhdeta(t,k)/2 ), y[3/7] = Im( sqrt(2/k)*a(t)*dhdeta(t,k)/2 ), eta being conformal time, a*deta = dt
    k : float
        the comoving wavenumber in Hubble units
    H : float
        the Hubble rate at time t (in Hubble units)
    a : float
        the scalefactor at time t

    Returns
    -------
    dydt : array
        an array of time derivatives of y

    """
    dydt = np.zeros(y.shape)

    #real
    dydt[0] = (H*y[0] + k/a*y[1])
    dydt[1] = ( -H*y[1] - k/a*y[0] )

    #imaginary
    dydt[2] = (H*y[2] + k/a*y[3])
    dydt[3] = ( -H*y[3] - k/a*y[2] )

    return dydt

def GreenEoM(A : ArrayLike, k : float, H : float, a : float):
    """
    For fixed times t and t', and comoving wavenumber k (in Hubble units), compute the t' derivative of 
    B(k, t, t') = k*G(k, t, t') = k 1/2 a(t')^2 Im ( h(k, t) h^*(k, t') )
    and 
    C(k, t, t') = 1/2 a(t')^2 Im ( dhdeta(k, t) h^*(k, t') ) 
    Here, G(k, t, t') is the retarded Green function associated with the differential operator D = d/deta^2 + 2 H a d/deta + k^2

    Parameters
    ----------
    A : array
        the values of B(k, t, t') and C(k, t, t')
    k : float
        the comoving wavenumber in Hubble units
    H : float
        the Hubble rate at time t (in Hubble units)
    a : float
        the scalefactor at time t

    Returns
    -------
    dAdt : array
        an array of time derivatives of B(k, t, t') and C(k, t, t')

    """
    dAdt = np.zeros(A.shape)
    dAdt[0] =(2*H*A[0] + k/a*A[1])
    dAdt[1] = -k/a*A[0]
    return dAdt

class PowSpecT:
    """
    A class used to compute the gravitational-wave power spectrum, including vacuum and gauge-field induced contributions.
    This computation is based on knowledge of the background dynamics of axion inflation using the GEF and a set of gauge-field spectra.
    All quantities throughout the code are treated in Hubble units.
    
    ...
    
    Attributes
    ----------
    
    
    x.__t : array
        An increasing array of physical times tracking the evolution of the GEF system.
    x.__N : array
        An increasing array of e-Folds tracking the evolution of the GEF system.
    x.__H :  array
        An array of Hubble rates as a function of time.
    x.__xi : array
        An array of xi values as a function of time.
    x.__beta : float
        The strength of the inflaton--gauge-field interaction, beta/M_P
    x.__af : function
        returns the scale factor, a(t), as a function of physical time. Obtained by interpolation of the GEF solution.
    x.__Hf : function
        returns the Hubble rate, H(t), as a function of physical time. Obtained by interpolation of the GEF solution.
    x.__HN : function
        returns the Hubble rate, H(N), as a function of e-folds. Obtained by interpolation of the GEF solution.
    x.__khN : function
        returns the instability scale k_h(N) as a function of e-folds. Obtained by interpolation of the GEF solution.
    x.__etaf : function
        returns the conformal time eta(t) as a function of physical time normalised to eta(0)=-1/H_0. Obtained by numerical integration and interpolation.
    x.maxk : float
        the maximal comoving wavenumber k which can be resolved based on the dynamical range covered by the GEF solution
    x.mink : float
        the minimal comoving wavenumber k which can be resolved based on the initial conditions of the GEF solution
    x.__omega : float
        The ratio H_0/M_pl where H_0 is the value of the Hubble parameter at initialisation of the GEF system.
        Used to obtain gravitational-wave power spectra a a function of frequency today.
    x.maxN : float
        If the GEF solution captures the end of Inflation, contains the number of e-folds after initialisation corresponding to the end of inflation.
        Otherwise, contains the largest number of e-folds after initialisiation which is captured by the GEF.
        This value is used to determine the redshift of frequencies and the gravitational wave power spectrum. 
        This is the default value at which the tensor power spectrum is computed.

    ...
    
    Methods
    -------
    
    _InitialKTN_()
        Determines the solution to k = 10^(5/2)*a(t)H(t). 
        Initial data can be given for the comoving wavenumber k, the physical time coordinates t, or e-Folds N.
    _GetHomSol_()
        For a given comoving wavenumber k satisfying k=10^(5/2)*a(t)H(t), initialises the vacuum tensor modes at time t in the Bunch-Davies vacuum
        and computes the time evolution within a given time interval, teval.
    _GreenFunc_()
        Computes the Green function G_k(N, N') for a given comoving wavenumber k at a given moment of time N for a range of times N' (time in e-folds)
    _VacuumPowSpec_()
        Computes the vacuum contribution to the tensor power spectrum for a given comoving wavenumber k
    _InducedTensorPowSpec_()
        Computes the gauge-field-induced power spectrum for a given comoving wavenumber k at a given moment of time N (in e-folds)
    ComputePowSpec()
        Computes the full tensor power spectrum (including vacuum and sourced contributions) for a specified range of comoving wavenumbers k.
    ktofreq()
        Red-shifts a comoving wavenumber k to obtain the corresponding requency in Hz today.
        Assumes  x.maxN corresponds to the end of inflation. 
    PTtoOmega():
        Converts a tensor power spectrum to the gravitational-wave energy density, h^2 OmegaGW.
        Assumes  x.maxN corresponds to the end of inflation. 
    PTAnalyitcal():
        From a given GEF result, compute the analytical estimate of the tensor power spectrum from axion inflation.
    """
    def __init__(x, G):
        #Set GEF results to Hubble units.
        if G.units: G.Unitless()
        
        a = G.vals["a"]
        H = G.vals["H"]
        
        Nend = G.EndOfInflation()[0]
        N = G.vals["N"]

        x.__omega = G.omega
        
        #Assess if the end of inflation is reached for this run
        if max(N) < Nend:
            print("This GEF run has not run reached the end of inflation. The code will assume Nend = max(N). Proceed with caution!")
        maxN = min(max(N), Nend)
        x.maxN = maxN
            
        #Set the range of modes
        x.maxk = CubicSpline(N, a*H)(maxN)
        x.mink = 1e4

        #Define Useful quantities
        x.__beta = G.beta

        x.__t = G.vals["t"]
        x.__N = N
        x.__H = H
        x.__xi= G.vals["xi"]

        
        x.__af = CubicSpline(x.__t, a)
        x.__Hf = CubicSpline(x.__t, H)
        x.__HN = CubicSpline(x.__N, x.__H)
        x.__khN = CubicSpline(N, G.vals["kh"])

        #Obtain eta as a functio of time
        deta = lambda t, y: 1/x.__af(t)
        
        soleta = solve_ivp(deta, [min(x.__t), max(x.__t)], np.array([-1]), t_eval=x.__t)

        x.__etaf = CubicSpline(x.__t, soleta.y[0,:])
        return
    
    def _InitialKTN_(x, init : ArrayLike, mode : str ="t", pwr : float=5/2):
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
            an array of comoving wavenumbers k satisfying k=10^(5/2) a(tstart)H(tstart)
        tstart : array
            an array of physical time coordinates t satisfying k=10^(5/2) a(tstart)H(tstart)
        """

        t = x.__t
        logkH = lambda t: np.log(x.__af(t)*x.__Hf(t))
        if mode=="t":
            tstart = init
            logks = logkH(tstart)
            k = 10**(pwr)*np.exp(logks)

        elif mode=="k":
            k = init
            x0 = np.log(k[0]) - 5/2*np.log(10)
            tstart = []
            for i, l in enumerate(k):
                f = lambda x: np.log(l) - logkH(x) - pwr*np.log(10)
                ttmp = fsolve(f, x0)[0]
                #Update the initial guess based on the previous result
                if i < len(k)-1:
                    x0 = ttmp + np.log(k[i+1]/l)
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(x.__N, t)(init)
            k = 10**pwr*np.exp(logkH(tstart))

        else:
            print("not a valid choice")
            raise KeyError

        return k, tstart
        
    def _GetHomSol_(x, k : float, tstart : float, teval : ArrayLike|list=[], atol : float=1e-3, rtol : float=1e-4):
        """
        Input
        -----
        k : float
           the comoving wavenumber k for which the mode function h(t,k) is evolved.
        tstart : float
            the time coordinate satisfying k = 10^(5/2)k_h(tstart) needed to ensure that the modes initialised in the Bunch-Davies vacuum
        teval : array|list
            physical time points at which the tensor mode function and its derivatives will be returned.
            If teval=[], the mode functions are evaluated at x.__t
        atol : float
            the absolute precision of the numerical intergrator
        rtol : float
            the relative precision of the numerical integrator

        Return
        ------
        phik : array
            the vacuum tensor mode (rescaled), sqrt(2k)*h(teval, k)/2
        dphik : array
            the derivative of the vacuum tensor mode (rescaled), sqrt(2/k)*dhdeta(teval, k)/2
        """

        if len(teval)==0:
            teval = x.__t

        #conformal time needed for relative phases
        eta = x.__etaf(teval)

        tend = max(x.__t)
        istart = 0
        while teval[istart]<tstart:
            istart+=1
        
        #define the ODE for the GW modes
        ode = lambda t, y: TensorModeEoM( y, k, x.__Hf(t), x.__af(t) )
        
        #Initialise the modes in Bunch Davies
        Zini = np.array([1, -10**(-5/2), 0, -1])

        sol = solve_ivp(ode, [tstart, tend], Zini, t_eval=teval[istart:], method="RK45", atol=atol, rtol=rtol)
        if not(sol.success):
            print("Something went wrong")

        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create an array tracking a modes evolution from Bunch Davies to late times. Ensure equal length arrays for every mode k
        phik = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )/x.__af(teval)
        dphik = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )/x.__af(teval)

        return phik, dphik

    def _GreenFunc_(x, k : float, phik : ArrayLike, ind : int, tstart : float, teval : ArrayLike|list=[], atol : float=1e-3, rtol : float=1e-4):
        """
        Input
        -----
        k : float
            the comoving wavenumber k for which the retarded Green function G(k, t, t') = Im( h(k, t) h^*(k, t') ) / Im( dhdeta(k, t') h^*(k, t') ) is evolved.
        phik : array
            the values of the vacuum tensor mode phi(teval, k) = sqrt(2k)*h(teval, k)/2
        tstart : float
            the time coordinate satisfying k = 10^(5/2)k_h(tstart), s.t. phi(t <= tstart, k) is in the Bunch-Davies state.
        ind : integer
            The fixed time t in G(k, t, t') is given by t = teval[ind]
        teval : array|list
            physical time points at which the Green function is returned.
            If teval=[], the Green function is evaluated at x.__t
            teval must coincide with the times at which phik is evaluated!
        atol : float
            the absolute precision of the numerical intergrator 
        rtol : float
            the relative precision of the numerical integrator

        Return
        ------
        GreenN : array
            the values of B(k, teval[ind], teval) = - k G(k, teval[ind], teval) 
        """
        if len(teval)==0:
            teval = x.__t
            
        istart = 0
        while teval[istart]<tstart:
            istart+=1

        # G(k, t, t) = 0 by definition
        Aini = np.array([0, 1])

        # Solve the EoM for G backwards in time starting from G(k, t, t)
        Aode = lambda t, y: -GreenEoM(y, k, x.__Hf(-t), x.__af(-t))
        solA = solve_ivp(Aode, [-teval[ind], -tstart], Aini, t_eval=-teval[istart:ind+1][::-1],
                         method="RK45", atol=atol, rtol=rtol)

        #For numerical stability, only solve the EoM for the Green function until t' = tstart. Afterwards, compute it directly from the vacuum modes.
        GreenN = np.zeros(teval.shape)
        GreenN[istart:ind+1] = solA.y[0,:][::-1]
        GreenN[:istart] = ( (phik[ind].conjugate()*phik).imag*x.__af(teval)**2 )[:istart]

        return GreenN

    def _VacuumPowSpec_(x, k : ArrayLike, phik : ArrayLike):
        """
        Input
        -----
        k : array
            an array of comoving wavenumbers k for which the vacuum power spectrum is computed
        phik : array
            the values of the vacuum tensor mode phi(t, k) = sqrt(2k)*h(teval, k)/2

        Return
        ------
        PTvac : array
            the vacuum tensor power spectrum PT_vac(t, k) at a fixed time t as a function of comoving wavenumber.
        """
        PTvac = 2*(k*x.__omega)**2/( np.pi**2 ) * abs(phik)**2
        return PTvac

    def _InducedTensorPowerSpec_(x, k : float, lgrav : float, ind: int, Ngrid : ArrayLike, GreenN : ArrayLike, kgrid : ArrayLike,
                                l1 : float, A1 : ArrayLike, dA1 : ArrayLike, l2 : float, A2 : ArrayLike, dA2 : ArrayLike,
                                momgrid : int=100):
        """
        Input
        -----
        k : float
            the comoving wavenumber k for which the gauge-field induced power spectrum is computed.
        lgrav : array
            the gravitational-wave helicity for which the gauge-field induced power spectrum is computed.
        ind : integer
            the power spectrum is computed at a time N = Ngrid[ind] (in e-folds)
        Ngrid : array
            an array of e-folds over which the Green function and gauge-mode functions are integrated
        GreenN : array
            a 1D-array of Green function values k*G(k, Ngrid[ind], Ngrid)
        kgrid : array
            an array of comoving wavenumbers for which the gauge-field mode functions are given.
        l1, l2 : float
            the gauge-field helicities of the given mode functions
        A1, A2 : float
            the gauge-field mode functions sqrt(2k)*A(Ngrid, kgrid, l1/l2) as obtained from the mode-by-mode code
        dA1, dA2 : float
            the derivatives of the gauge-field mode functions sqrt(2/k)*dAdeta(Ngrid, kgrid, l1/l2) as obtained from the mode-by-mode code
        momgrid : int
            the internal momentum integral over p and k-p is performed using a momgrid x momgrid grid.

        Return
        ------
        PTind : float
            the gauge-field induced tensor power spectrum PT_ind(Ngrid[ind], k, lgrav).
        """

        cutUV = x.__khN(Ngrid[ind])/k
        cutIR = min(kgrid)/k
        HN = x.__HN(Ngrid)

        logAs = np.linspace(np.log(max(0.5, cutIR)), np.log(cutUV), momgrid)

        Afuncx = CubicSpline(np.log(kgrid), A1)
        dAfuncx = CubicSpline(np.log(kgrid), dA1)
        
        Afuncy = CubicSpline(np.log(kgrid), A2)
        dAfuncy = CubicSpline(np.log(kgrid), dA2)

        IntOuter = []
        for logA in logAs:
            A = np.exp(logA)
            Blow = (cutIR - A)
            Bhigh = (A - cutIR)
            if Bhigh>0.5:
                Blow = (A - cutUV)
                Bhigh = (cutUV - A)
                if Bhigh>0.5:
                    Blow = -0.5
                    Bhigh = 0.5
            Bs = np.linspace(Blow, Bhigh, momgrid)[1:-1]

            IntInner = np.zeros(Bs.shape)

            for j, B in enumerate(Bs):
                Ax = Afuncx(np.log(k*(A+B)))
                dAx = dAfuncx(np.log(k*(A+B)))
                Ay = Afuncy(np.log(k*(A-B)))
                dAy = dAfuncy(np.log(k*(A-B)))

                mom =  abs( l1*l2 + 2*lgrav*( (l1+l2)*A + (l1-l2)*B ) + 4*(A**2 - B**2) + 8*lgrav*A*B*( (l1-l2)*A - (l1+l2)*B ) - 16*l1*l2*A**2*B**2 )
                z = max(A+B,A-B)
                mask = np.where(z<x.__khN(Ngrid)/k, 1, 0)

                val = (dAx*dAy + l1*l2*Ax*Ay)*mask*k/(np.exp(3*Ngrid)*HN)
                timeintegrand = GreenN*val*mom
            
                timeintre = trapezoid(timeintegrand[:ind].real, Ngrid[:ind])
                
                timeintim = trapezoid(timeintegrand[:ind].imag, Ngrid[:ind])
                
                IntInner[j] = (timeintre**2 + timeintim**2)*A

  
            IntOuter.append(trapezoid(IntInner, Bs))
        IntOuter = np.array(IntOuter)

        PTind = trapezoid(IntOuter, logAs) / (16*np.pi**4)*(k*x.__omega)**4

        return PTind

    
    def ComputePowSpec(x, k : ArrayLike, N : float|None=None, ModePath : str|None=None, FastGW : bool=True,
                       atol : float=1e-3, rtol : float=1e-4, momgrid : int=100):
        """
        Input
        -----
        k : array
            the comoving wavenumber k for which to compute the tensor power spectrum.
        N : float|None
            the time (in e-folds) at which to compute the tensor power spectrun.
            If N=None, the tensor power spectrum is computed at x.maxN.
        ModePath : str
            The path to a file containing the tabulated gauge-field mode functions from a mode-by-mode computation.
        FastGW : bool
            If FastGW = True, only those contributions to the induce power spectrum coming from the most amplified mode-functions is computed.
            I.e., if xi>0, then only the PT_ind^+/i(++) contributions are computed.
            Else, compute all contributions to the power spectrum.
        atol : float
            the absolute precision of the numerical intergrator used to compute the vacuum tensor mode functions and the Green function.
        rtol : float
             the relative precision of the numerical intergrator used to compute the vacuum tensor mode functions and the Green function.
        momgrid : int
            the internal momentum integral for the induced power spectrum over p and k-p is performed using a momgrid x momgrid grid

        Return
        ------
        PT : dict
            a dictionary containing all contributions to the tensor power spectrum, including the total power spectrum.
        """

        ks, tstarts = x._InitialKTN_(k, mode="k")

        if ModePath==None:
            ModePath = f"../Modes/Modes+Beta{x.__beta}+M6_16.dat"
        
        tgrid, Ngrid, kgrid, Ap, dAp, Am, dAm = ReadMode(ModePath)
        Ngrid = np.array(list(Ngrid))

        GaugeModes = {"+":(Ap, dAp), "-":(Am, dAm)}

        if N==None:
            N = x.maxN
        
        inds = np.where(Ngrid < N)[0]
        indend = inds[-1]

        PT = {"tot":[], "vac":[], "ind+,++":[], "ind+,+-":[], "ind+,--":[], "ind-,++":[], "ind-,+-":[], "ind-,--":[]}

        GWpols = [("+", 1), ("-",-1)]

        gaugepols=[(("+",1),("+",1)),
                    (("+",1),("-",-1)),
                        (("-",-1),("-",-1))]

        sign = np.sign(x.__xi[0])

        for i, k in enumerate(ks):
            tstart = tstarts[i]

            if k > 5*(x.__af(tgrid[indend])*x.__Hf(tgrid[indend])):
                for key in PT.keys():
                    PT[key].append(0)
            else:
                f, _ = x._GetHomSol_(k, tstart, tgrid, atol=atol, rtol=rtol)
                Green = x._GreenFunc_(k, f, indend, tstart, tgrid, atol=atol, rtol=rtol)
                
                PT["vac"].append(x._VacuumPowSpec_(k, f[indend]))

                for lgrav in GWpols:
                    for mu in gaugepols:
                        GWpol = lgrav[1]

                        Ax = GaugeModes[mu[0][0]][0]
                        dAx = GaugeModes[mu[0][0]][1]
                        lx = mu[0][1]
                        Ay = GaugeModes[mu[1][0]][0]
                        dAy = GaugeModes[mu[1][0]][1]
                        ly = mu[1][1]

                        if FastGW and (lx != sign or ly != sign):
                            PT[f"ind{lgrav[0]},{mu[0][0]}{mu[1][0]}"].append(0.)
                        else:
                            PT[f"ind{lgrav[0]},{mu[0][0]}{mu[1][0]}"].append(
                                x._InducedTensorPowerSpec_(k, GWpol, indend, Ngrid, Green, kgrid, lx, Ax, dAx, ly, Ay, dAy, momgrid) )

        PT["tot"] = np.zeros(ks.shape)
        for key in PT.keys():
            PT[key] = np.array(PT[key])
            if ("+" in key) or ("-" in key):
                PT["tot"] += 0.5*PT[key]
            elif key=="vac":
                PT["tot"] += PT[key]

        return PT
    
    def PTAnalytical(x):
        """
        Return
        ------
        PT : dict
            a dictionary containing the most important contributions to the analytic tensor power spectrum, including the total power spectrum.
        """
        H = x.__omega*x.__H
        xi = abs(x.__xi)
        pre = (H/np.pi)**2 # * (x.__H)**(x.__nT)
        if np.sign(x.__xi[0]) > 0:                
            indP = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
        else:
            indP = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
        
        #Factors of two to match my convention
        PTanalytic = {"tot":(2*pre + indP + indM), "vac":2*pre, "ind+":2*indP, "ind-":2*indM}
        return PTanalytic
    
    def ktofreq(x, k : ArrayLike, Nend : float|None=None, DeltaN : float=0.):
        """
        Input
        -----
        k : array
            an array of comoving wavenumbers k during inflation
        Nend : float
            the number of e-folds corresponding to the end of inflation.
        DeltaN : float
            an uncertainty of e-folds encoding the duration of reheating. Instantaneous reheating assumes deltaN=0

        Return
        ------
        f : array
            the red-shifted frequencies in Hz
        """
        if Nend==None:
            Nend = x.maxN

        #We assume instantaneous reheating. To parametrise this assumption, shift the output frequency f -> f exp(- Nrh). Nrh is the unknown number of e-folds of reheating.
        Hrh = x.__HN(Nend)

        Trh = np.sqrt(3*Hrh*x.__omega/np.pi)*(10/106.75)**(1/4)*M_pl
        Trh = Trh*(106.75/g_rho(Trh))**(1/4)

        f = k*x.__omega*M_pl*gev_to_hz/(2*np.pi*np.exp(Nend)) * T_0/Trh * (g_s(Trh)/g_s_0)**(-1/3)*np.exp(-DeltaN)

        return f

    def PTtoOmega(x, PT : ArrayLike, k : ArrayLike, Nend : float|None=None, DeltaN : float=0.):
        """
        Input
        -----
        PT : array
            an array of tensor power spectra at the end of inflation for comoving wavenumbers k
        k : array
            an array of comoving wavenumbers k during inflation
        Nend : float
            the number of e-folds corresponding to the end of inflation.
        DeltaN : float
            an uncertainty of e-folds encoding the duration of reheating. Instantaneous reheating assumes DeltaN=0

        Return
        ------
        f : array
            the red-shifted frequencies in Hz
        """
        f = x.ktofreq(k, Nend, DeltaN)
        OmegaGW = h**2*omega_r/24  * PT * (g_rho(f, True)/g_rho_0) * (g_s_0/g_s(f, True))**(4/3)
        return OmegaGW, f
    


