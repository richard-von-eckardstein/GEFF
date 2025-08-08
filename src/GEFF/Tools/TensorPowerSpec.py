import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapezoid

from GEFF.ModeByMode import ReadMode

from typing import Tuple
from numpy.typing import ArrayLike


def TensorModeEoM(y : ArrayLike, k : float, H : float, a : float) -> ArrayLike:
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

def GreenEoM(A : ArrayLike, k : float, H : float, a : float) -> ArrayLike:
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
    
    
    self.__t : array
        An increasing array of physical times tracking the evolution of the GEF system.
    self.__N : array
        An increasing array of e-Folds tracking the evolution of the GEF system.
    self.__H :  array
        An array of Hubble rates as a function of time.
    self.__xi : array
        An array of xi values as a function of time.
    self.__beta : float
        The strength of the inflaton--gauge-field interaction, beta/M_P
    self.__af : function
        returns the scale factor, a(t), as a function of physical time. Obtained by interpolation of the GEF solution.
    self.__Hf : function
        returns the Hubble rate, H(t), as a function of physical time. Obtained by interpolation of the GEF solution.
    self.__HN : function
        returns the Hubble rate, H(N), as a function of e-folds. Obtained by interpolation of the GEF solution.
    self.__khN : function
        returns the instability scale k_h(N) as a function of e-folds. Obtained by interpolation of the GEF solution.
    self.__etaf : function
        returns the conformal time eta(t) as a function of physical time normalised to eta(0)=-1/H_0. Obtained by numerical integration and interpolation.
    self.maxk : float
        the maximal comoving wavenumber k which can be resolved based on the dynamical range covered by the GEF solution
    self.mink : float
        the minimal comoving wavenumber k which can be resolved based on the initial conditions of the GEF solution
    self.__omega : float
        The ratio H_0/M_pl where H_0 is the value of the Hubble parameter at initialisation of the GEF system.
        Used to obtain gravitational-wave power spectra a a function of frequency today.
    self.maxN : float
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
        Assumes  self.maxN corresponds to the end of inflation. 
    PTtoOmega():
        Converts a tensor power spectrum to the gravitational-wave energy density, h^2 OmegaGW.
        Assumes  self.maxN corresponds to the end of inflation. 
    PTAnalyitcal():
        From a given GEF result, compute the analytical estimate of the tensor power spectrum from axion inflation.
    """
    def __init__(self, values):
        #Set GEF results to Hubble units.
        values.SetUnits(False)
        
        a = values.a
        H = values.H
        
        Nend = values.N[-1]
        N = values.N

        self.__omega = values.H0
        
        #Assess if the end of inflation is reached for this run
        """if np.log10(abs(max(N) - Nend)) > -2:
            print("This GEF run has not run reached the end of inflation. The code will assume Nend = max(N). Proceed with caution!")"""
        maxN = min(max(N), Nend)
        self.maxN = maxN
            
        #Set the range of modes
        self.maxk = CubicSpline(N, a*H)(maxN)
        self.mink = 1e4

        #Define Useful quantities

        self.__t = values.t
        self.__N = N
        self.__H = H
        self.__xi= values.xi

        
        self.__af = CubicSpline(self.__t, a)
        self.__Hf = CubicSpline(self.__t, H)
        self.__HN = CubicSpline(self.__N, self.__H)
        self.__khN = CubicSpline(N, values.kh)

        #Obtain eta as a functio of time
        deta = lambda t, y: 1/self.__af(t)
        
        soleta = solve_ivp(deta, [min(self.__t), max(self.__t)], np.array([-1]), t_eval=self.__t)

        self.__etaf = CubicSpline(self.__t, soleta.y[0,:])
        return
    
    def _InitialKTN_(self, init : ArrayLike, mode : str ="t", pwr : float=5/2) -> Tuple[ArrayLike, ArrayLike]:
        """
        Input
        -----
        init : array
           an array of physical time coordinates t, OR of e-Folds N, OR of comoving wavenumbers k (within self.mink and self.maxk)
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

        t = self.__t
        logkH = lambda t: np.log(self.__af(t)*self.__Hf(t))
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
            tstart = CubicSpline(self.__N, t)(init)
            k = 10**pwr*np.exp(logkH(tstart))

        else:
            print("not a valid choice")
            raise KeyError

        return k, tstart
        
    def _GetHomSol_(self, k : float, tstart : float, teval : ArrayLike|list=[], atol : float=1e-3, rtol : float=1e-4) -> Tuple[ArrayLike, ArrayLike]:
        """
        Input
        -----
        k : float
           the comoving wavenumber k for which the mode function h(t,k) is evolved.
        tstart : float
            the time coordinate satisfying k = 10^(5/2)k_h(tstart) needed to ensure that the modes initialised in the Bunch-Davies vacuum
        teval : array|list
            physical time points at which the tensor mode function and its derivatives will be returned.
            If teval=[], the mode functions are evaluated at self.__t
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
            teval = self.__t
        tend = max(teval)

        #conformal time needed for relative phases
        eta = self.__etaf(teval)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        
        #define the ODE for the GW modes
        ode = lambda t, y: TensorModeEoM( y, k, self.__Hf(t), self.__af(t) )
        
        #Initialise the modes in Bunch Davies
        Zini = np.array([1, -10**(-5/2), 0, -1])

        sol = solve_ivp(ode, [tstart, tend], Zini, t_eval=teval[istart:], method="RK45", atol=atol, rtol=rtol)
        if not(sol.success):
            print("Something went wrong")

        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create an array tracking a modes evolution from Bunch Davies to late times. Ensure equal length arrays for every mode k
        phik = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )/self.__af(teval)
        dphik = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )/self.__af(teval)

        return phik, dphik

    def _GreenFunc_(self, k : float, phik : ArrayLike, ind : int, tstart : float, teval : ArrayLike|list=[], atol : float=1e-30, rtol : float=1e-4) -> ArrayLike:
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
            If teval=[], the Green function is evaluated at self.__t
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
            teval = self.__t
            
        istart = 0
        while teval[istart]<tstart:
            istart+=1

        # G(k, t, t) = 0 by definition
        Aini = np.array([0, 1])
    

        # Solve the EoM for G backwards in time starting from G(k, t, t)
        Aode = lambda t, y: -GreenEoM(y, k, self.__Hf(-t), self.__af(-t))
        solA = solve_ivp(Aode, [-teval[ind], -tstart], Aini, t_eval=-teval[istart:ind+1][::-1],
                         method="RK45", atol=atol, rtol=rtol)

        #For numerical stability, only solve the EoM for the Green function until t' = tstart. Afterwards, compute it directly from the vacuum modes.
        GreenN = np.zeros(teval.shape)
        GreenN[istart:ind+1] = solA.y[0,:][::-1]
        GreenN[:istart] = ( (phik[ind].conjugate()*phik).imag*self.__af(teval)**2 )[:istart]

        return GreenN

    def _VacuumPowSpec_(self, k : ArrayLike, phik : ArrayLike) -> ArrayLike:
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
        PTvac = 2*(k*self.__omega)**2/( np.pi**2 ) * abs(phik)**2
        return PTvac

    def _InducedTensorPowerSpec_(self, k : float, lgrav : float, ind: int, Ngrid : ArrayLike, GreenN : ArrayLike, kgrid : ArrayLike,
                                l1 : float, A1 : ArrayLike, dA1 : ArrayLike, l2 : float, A2 : ArrayLike, dA2 : ArrayLike,
                                momgrid : int=100) -> float:
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

        cutUV = self.__khN(Ngrid[ind])/k
        cutIR = min(kgrid)/k
        HN = self.__HN(Ngrid)

        logAs = np.linspace(np.log(max(0.5, cutIR)), np.log(cutUV), momgrid)

        #Alternatives for interpolating mode functions directly? Could be problematic as they are highly oscillatory
        #It may be better to directly interpolate the integrand...
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
                mask = np.where(z<self.__khN(Ngrid)/k, 1, 0)

                val = (dAx*dAy + l1*l2*Ax*Ay)*mask*k/(np.exp(3*Ngrid)*HN)
                timeintegrand = GreenN*val*mom
            
                timeintre = trapezoid(timeintegrand[:ind].real, Ngrid[:ind])
                
                timeintim = trapezoid(timeintegrand[:ind].imag, Ngrid[:ind])
                
                IntInner[j] = (timeintre**2 + timeintim**2)*A

  
            IntOuter.append(trapezoid(IntInner, Bs))
        IntOuter = np.array(IntOuter)

        PTind = trapezoid(IntOuter, logAs) / (16*np.pi**4)*(k*self.__omega)**4

        return PTind

    
    def ComputePowSpec(
                        self, nmodes : int, N : float|None=None, ModePath : str|None=None, FastGW : bool=True,
                       atols : list=[1e-3,1e-20], rtols : list=[1e-4,1e-4], momgrid : int=100
                       ) -> Tuple[ArrayLike,dict]:
        """
        Input
        -----
        k : array
            the comoving wavenumber k for which to compute the tensor power spectrum.
        N : float|None
            the time (in e-folds) at which to compute the tensor power spectrun.
            If N=None, the tensor power spectrum is computed at self.maxN.
        ModePath : str
            The path to a file containing the tabulated gauge-field mode functions from a mode-by-mode computation.
        FastGW : bool
            If FastGW = True, only those contributions to the induce power spectrum coming from the most amplified mode-functions is computed.
            I.e., if xi>0, then only the PT_ind^+/i(++) contributions are computed.
            Else, compute all contributions to the power spectrum.
        atols : list
            the absolute precision of the numerical intergrator used to compute the vacuum tensor mode functions (index 0) and the Green function (index 1).
        rtols : list
             the relative precision of the numerical intergrator used to compute the vacuum tensor mode functions (index 0) and the Green function (index 1).
        momgrid : int
            the internal momentum integral for the induced power spectrum over p and k-p is performed using a momgrid x momgrid grid

        Return
        ------
        PT : dict
            a dictionary containing all contributions to the tensor power spectrum, including the total power spectrum.
        """

        k = np.logspace(np.log10(self.mink), np.log10(10*self.maxk), nmodes)
        ks, tstarts = self._InitialKTN_(k, mode="k")
        
        spec = ReadMode(ModePath)
        Ngrid = spec["N"]
        tgrid = spec["t"]
        kgrid = spec["k"]

        GaugeModes = {"+":(spec["Ap"], spec["dAp"]), "-":(spec["Am"], spec["dAm"])}

        if N==None:
            N = self.maxN
        
        inds = np.where(Ngrid < N)[0]
        indend = inds[-1]

        PT = {"tot":[], "vac":[], "ind+,++":[], "ind+,+-":[], "ind+,--":[], "ind-,++":[], "ind-,+-":[], "ind-,--":[]}

        GWpols = [("+", 1), ("-",-1)]

        gaugepols=[(("+",1),("+",1)),
                    (("+",1),("-",-1)),
                        (("-",-1),("-",-1))]

        sign = np.sign(self.__xi[0])

        for i, k in enumerate(ks):
            tstart = tstarts[i]

            if k > 5*(self.__af(tgrid[indend])*self.__Hf(tgrid[indend])):
                for key in PT.keys():
                    PT[key].append(0)
            else:
                f, _ = self._GetHomSol_(k, tstart, tgrid, atol=atols[0], rtol=rtols[0])
                Green = self._GreenFunc_(k, f, indend, tstart, tgrid, atol=atols[1], rtol=rtols[1])
                
                PT["vac"].append(self._VacuumPowSpec_(k, f[indend]))

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
                                self._InducedTensorPowerSpec_(k, GWpol, indend, Ngrid, Green, kgrid, lx, Ax, dAx, ly, Ay, dAy, momgrid) )

        PT["tot"] = np.zeros(ks.shape)
        for key in PT.keys():
            PT[key] = np.array(PT[key])
            if ("+" in key) or ("-" in key):
                PT["tot"] += 0.5*PT[key]
            elif key=="vac":
                PT["tot"] += PT[key]

        return ks, PT
    
    def PTAnalytical(self) -> Tuple[ArrayLike,dict]:
        """
        Return
        ------
        PT : dict
            a dictionary containing the most important contributions to the analytic tensor power spectrum, including the total power spectrum.
        """
        H = self.__omega*self.__H
        xi = abs(self.__xi)
        pre = (H/np.pi)**2 # * (self.__H)**(self.__nT)
        if np.sign(self.__xi[0]) > 0:                
            indP = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
        else:
            indP = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
        
        #Factors of two to match my convention
        PTanalytic = {"tot":(2*pre + indP + indM), "vac":2*pre, "ind+":2*indP, "ind-":2*indM}
        k = H*np.exp(self.__N)
        return k, PTanalytic
    


