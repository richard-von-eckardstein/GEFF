r"""
This module is designed to compute the tensor power spectrum from a GEF result.

The polarized power spectrum at time $t$ with momentum $k$ and helicity $\lambda$ is defined as
$$\mathcal{P}_{T,\lambda}(t, k) = \mathcal{P}_{T,\lambda}^{\mathrm{vac}}(t, k) + \mathcal{P}_{T,\lambda}^{\mathrm{ind}}(t, k)\, ,$$
with the vacuum contribution
$$\mathcal{P}_{T,\lambda}^{\mathrm{vac}}(t, k) = \frac{4 k^3}{\pi^2 M_{\rm P}^2} |u_0(t, k)|^2\, ,$$
and induced contribution
$$\mathcal{P}_{T,\lambda}^{\mathrm{ind}}(t, k) = \frac{k^3}{2 \pi^2 M_{\rm P}^4} \int \frac{{\rm d}^3 {\bf p}}{(2 \pi)^3} \sum_{\alpha,\beta = \pm1} 
        \left(1 +  \lambda \alpha \frac{{\bf k} \cdot {\bf p}}{k p} \right)^2 \left(1 +  \lambda \beta \frac{k^2 - {\bf k} \cdot {\bf p}}{kq}  \right)^2 $$
$$ \qquad \qquad \qquad \times \left|\int_{-\infty}^\infty {\rm d} s \frac{G_k(t, s)}{a^3(s)} 
    \left[A'_\alpha(s, p)A'_\beta(s, q) + \alpha \beta\, p q\, A_\alpha(s, p) A_\alpha(s, q) \right] \right|^2 \, .
$$
with momentum $q = |{\bf p} + {\bf q}|$ and scale-factor $a$.

The vacuum modes $u_0(t,k)$ obey the mode equation
$$\mathcal{D}_k {u_0} = \ddot{u}_0 + 3 H \dot{u}_0 + \frac{k^2}{a^2} {u_0} = 0$$
with the retarded Green function $G_k(t',t)$ defined for the operator $\mathcal{D}_k$.

The gauge-field mode functions $A_\lambda(t,k)$ are defined as in the `GEFF.mode_by_mode` module.

For details on the numerical computation, see the Appendix B of [2508.00798](https://arxiv.org/abs/2508.00798).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapezoid

from GEFF.mode_by_mode import GaugeSpec
from GEFF.bgtypes import BGSystem

from typing import Tuple


def tensor_mode_eq(y : np.ndarray, k : float, H : float, a : float) -> np.ndarray:
    r"""
    Mode equation for vacuum tensor modes.

    Parameters
    ----------
    y : array
        the vacuum tensor mode and its derivatives.
        $a \sqrt{2k} u_0(t,k)$ and $ a^2\sqrt{2/k} \dot{u}_0(t, k)$ 
    k : float
        comoving momentum $k$
    H : float
        Hubble rate, $H(t)$
    a : float
        scale factor, $a(t)$

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

def green_ode(y : np.ndarray, k : float, H : float, a : float) -> np.ndarray:
    r"""
    Evolve the re-scaled Green function for tensor modes in time.

    The re-scaled Green function is
    $$B_{t'}^k(t) = 2 k a(t)^2 \operatorname{Im} \left[ u_0^*(t', k) \, u_0(t,k) \right] = k G_k(t', t) \,.$$
     
    whose ODE is coupled to 

    $$C_{t'}^k(t) = 2 a(t)^3 \operatorname{Im} \left[ u_0^*(t', k) \, {\dot{u}_0}(t,k) \right] \,.$$

    Parameters
    ----------
    y : array
        the functions B and C
    k : float
        comoving momentum $k$
    H : float
        Hubble rate, $H(t)$
    a : float
        scale factor, $a(t)$

    Returns
    -------
    dydt : array
        an array of time derivatives of A

    """
    dydt = np.zeros(y.shape)
    dydt[0] =(2*H*y[0] + k/a*y[1])
    dydt[1] = -k/a*y[0]
    return dydt

class PowSpecT:
    r"""
    A class used to compute the tensor power spectrum including vacuum and gauge-field induced contributions.

    These main method of this module is `ComputePowSpec`, which computes both the vacuum and gauge-field induced contribution
      to the tensor power spectrum, $\mathcal{P}_{T,\lambda}^{\mathrm{vac}}$ and $\mathcal{P}_{T,\lambda}^{\mathrm{ind}}$.

    Results are internally computed using numerical units, but are returned in physical units.
    """
    def __init__(self, sys : BGSystem):
        """
        Initialise the class from a GEF solution.

        Parameters:
        sys : BGSystem
            the GEF solution.
        """
        #Set GEF results to Hubble units.
        sys.set_units(False)
        
        #import the background evolution
        a = sys.a.value
        H = sys.H.value
        
        N = sys.N.value

        self._H0 = sys.H0
            
        #Set the range of modes
        self.maxk = np.maximum(a*H)
        self.mink = 1e4

        #Define Useful quantities

        self._t = sys.t
        self._N = N
        self._H = H
        self._xi= sys.xi

        #define interpolated quantities as needed
        self._af = CubicSpline(self._t, a)
        self._Hf = CubicSpline(self._t, H)
        self._HN = CubicSpline(self._N, self._H)
        self._khN = CubicSpline(N, sys.kh)

        #Obtain eta as a functio of time for phases
        def deta(t, y): return 1/self._af(t)
        
        soleta = solve_ivp(deta, [min(self._t), max(self._t)], np.array([0]), t_eval=self._t)

        self._etaf = CubicSpline(self._t, soleta.y[0,:])
        return
    
    def compute_pt(self, nmodes : int, mbm_file : str, FastGW : bool=True,
                    atols : list=[1e-3,1e-20], rtols : list=[1e-4,1e-4], ngrid : int=100
                    ) -> Tuple[np.ndarray,dict]:
        r"""
        Compute the full tensor power spectrum.

        The method lodes data on gauge-mode $A_\lambda(t,k)$ from a file indicated by `mbm_file`.
        The power spectrum is computed for `nmodes` log-spaced momenta $k$.

        Parameters
        ----------
        nmodes : array
            the number of momenta
        ModePath : str
            The path to a file containing the tabulated gauge-field mode functions from a mode-by-mode computation
        FastGW : bool
            If `True`, only the expected dominant contributions to $\mathcal{P}_{T,\lambda}^{\mathrm{ind}}$ are computed
        atols : list
            the absolute tolerance used by `_GetHomSol_` (index 0) and `_GreenFunc_` (index 1)
        rtols : list
            the relative tolerance used by `_GetHomSol_` (index 0) and `_GreenFunc_` (index 1)
        ngrid : int
            passed to `_InducedPowSpec_`
        Returns
        -------
        ks : NDArray
            the momenta $k$
        PT : dict
            a dictionary containing all contributions to the tensor power spectrum and its total.
        """

        k = np.logspace(np.log10(self.mink), np.log10(10*self.maxk), nmodes)
        ks, tstarts = self._InitialKTN_(k, mode="k")
        
        spec = GaugeSpec.read_spec(mbm_file)
        Ngrid = spec["N"]
        tgrid = spec["t"]
        kgrid = spec["k"]

        GaugeModes = {"+":(spec["Ap"], spec["dAp"]), "-":(spec["Am"], spec["dAm"])}
        
        inds = np.where(Ngrid < np.maximum(self._N))[0]
        indend = inds[-1]

        PT = {"tot":[], "vac":[], "ind+,++":[], "ind+,+-":[], "ind+,--":[], "ind-,++":[], "ind-,+-":[], "ind-,--":[]}

        GWpols = [("+", 1), ("-",-1)]

        gaugepols=[(("+",1),("+",1)),
                    (("+",1),("-",-1)),
                        (("-",-1),("-",-1))]

        sign = np.sign(self._xi[0])

        for i, k in enumerate(ks):
            tstart = tstarts[i]

            if k > 5*(self._af(tgrid[indend])*self._Hf(tgrid[indend])):
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
                                self._InducedTensorPowerSpec_(k, GWpol, indend, Ngrid, Green, kgrid, lx, Ax, dAx, ly, Ay, dAy, ngrid) )

        PT["tot"] = np.zeros(ks.shape)
        for key in PT.keys():
            PT[key] = np.array(PT[key])
            if ("+" in key) or ("-" in key):
                PT["tot"] += 0.5*PT[key]
            elif key=="vac":
                PT["tot"] += PT[key]

        return ks*self._H0, PT
    
    def _InitialKTN_(self, init : np.ndarray, mode : str ="t", pwr : float=5/2) -> Tuple[np.ndarray, np.ndarray]:
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

        t = self._t
        def logkH(t): return np.log(self._af(t)*self._Hf(t))
        if mode=="t":
            tstart = init
            logks = logkH(tstart)
            ks = 10**(pwr)*np.exp(logks)

        elif mode=="k":
            ks = init
            x0 = np.log(ks[0]) - 5/2*np.log(10)
            tstart = []
            for i, k in enumerate(ks):
                def f(x): return np.log(k) - logkH(x) - pwr*np.log(10)
                ttmp = fsolve(f, x0)[0]
                #Update the initial guess based on the previous result
                if i < len(ks)-1:
                    x0 = ttmp + np.log(ks[i+1]/k)
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(self._N, t)(init)
            ks = 10**pwr*np.exp(logkH(tstart))

        else:
            print("not a valid choice")
            raise KeyError

        return ks, tstart
        
    def _GetHomSol_(self, k : float, tstart : float, teval : np.ndarray|list=[], atol : float=1e-3, rtol : float=1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input
        -----
        k : float
           the comoving wavenumber k for which the mode function h(t,k) is evolved.
        tstart : float
            the time coordinate satisfying k = 10^(5/2)k_h(tstart) needed to ensure that the modes initialised in the Bunch-Davies vacuum
        teval : array|list
            physical time points at which the tensor mode function and its derivatives will be returned.
            If teval=[], the mode functions are evaluated at self._t
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
            teval = self._t
        tend = max(teval)

        #conformal time needed for relative phases
        eta = self._etaf(teval)

        istart = 0
        while teval[istart]<tstart:
            istart+=1
        
        #define the ODE for the GW modes
        def ode(t, y): return tensor_mode_eq( y, k, self._Hf(t), self._af(t) )
        
        #Initialise the modes in Bunch Davies
        Zini = np.array([1, -10**(-5/2), 0, -1])

        sol = solve_ivp(ode, [tstart, tend], Zini, t_eval=teval[istart:], method="RK45", atol=atol, rtol=rtol)
        if not(sol.success):
            print("Something went wrong")

        #the mode was in vacuum before tstart
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create an array tracking a modes evolution from Bunch Davies to late times. Ensure equal length arrays for every mode k
        phik = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )/self._af(teval)
        dphik = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )/self._af(teval)

        return phik, dphik

    def _GreenFunc_(self, k : float, phik : np.ndarray, ind : int, tstart : float, teval : np.ndarray|list=[], atol : float=1e-30, rtol : float=1e-4) -> np.ndarray:
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
            If teval=[], the Green function is evaluated at self._t
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
            teval = self._t
            
        istart = 0
        while teval[istart]<tstart:
            istart+=1

        # G(k, t, t) = 0 by definition
        Aini = np.array([0, 1])
    

        # Solve the EoM for G backwards in time starting from G(k, t, t)
        def Aode(t, y): return -green_ode(y, k, self._Hf(-t), self._af(-t))
        solA = solve_ivp(Aode, [-teval[ind], -tstart], Aini, t_eval=-teval[istart:ind+1][::-1],
                         method="RK45", atol=atol, rtol=rtol)

        #For numerical stability, only solve the EoM for the Green function until t' = tstart. Afterwards, compute it directly from the vacuum modes.
        GreenN = np.zeros(teval.shape)
        GreenN[istart:ind+1] = solA.y[0,:][::-1]
        GreenN[:istart] = ( (phik[ind].conjugate()*phik).imag*self._af(teval)**2 )[:istart]

        return GreenN

    def _VacuumPowSpec_(self, k : np.ndarray, phik : np.ndarray) -> np.ndarray:
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
        PTvac = 2*(k*self._H0)**2/( np.pi**2 ) * abs(phik)**2
        return PTvac

    def _InducedTensorPowerSpec_(self, k : float, lgrav : float, ind: int, Ngrid : np.ndarray, GreenN : np.ndarray, kgrid : np.ndarray,
                                l1 : float, A1 : np.ndarray, dA1 : np.ndarray, l2 : float, A2 : np.ndarray, dA2 : np.ndarray,
                                ngrid : int=100) -> float:
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
        ngrid : int
            the internal momentum integral over p and k-p is performed on a grid ngrid x ngrid grid.

        Return
        ------
        PTind : float
            the gauge-field induced tensor power spectrum PT_ind(Ngrid[ind], k, lgrav).
        """

        cutUV = self._khN(Ngrid[ind])/k
        cutIR = min(kgrid)/k
        HN = self._HN(Ngrid)

        logAs = np.linspace(np.log(max(0.5, cutIR)), np.log(cutUV), ngrid)

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
            Bs = np.linspace(Blow, Bhigh, ngrid)[1:-1]

            IntInner = np.zeros(Bs.shape)

            for j, B in enumerate(Bs):
                Ax = Afuncx(np.log(k*(A+B)))
                dAx = dAfuncx(np.log(k*(A+B)))
                Ay = Afuncy(np.log(k*(A-B)))
                dAy = dAfuncy(np.log(k*(A-B)))

                mom =  abs( l1*l2 + 2*lgrav*( (l1+l2)*A + (l1-l2)*B ) + 4*(A**2 - B**2) + 8*lgrav*A*B*( (l1-l2)*A - (l1+l2)*B ) - 16*l1*l2*A**2*B**2 )
                z = max(A+B,A-B)
                mask = np.where(z<self._khN(Ngrid)/k, 1, 0)

                val = (dAx*dAy + l1*l2*Ax*Ay)*mask*k/(np.exp(3*Ngrid)*HN)
                timeintegrand = GreenN*val*mom
            
                timeintre = trapezoid(timeintegrand[:ind].real, Ngrid[:ind])
                
                timeintim = trapezoid(timeintegrand[:ind].imag, Ngrid[:ind])
                
                IntInner[j] = (timeintre**2 + timeintim**2)*A

  
            IntOuter.append(trapezoid(IntInner, Bs))
        IntOuter = np.array(IntOuter)

        PTind = trapezoid(IntOuter, logAs) / (16*np.pi**4)*(k*self._H0)**4

        return PTind

    
    
    
    def PTAnalytical(self) -> Tuple[np.ndarray,dict]:
        """
        Return
        ------
        PT : dict
            a dictionary containing the most important contributions to the analytic tensor power spectrum, including the total power spectrum.
        """
        H = self._H0*self._H
        xi = abs(self._xi)
        pre = (H/np.pi)**2 # * (self._H)**(self._nT)
        if np.sign(self._xi[0]) > 0:                
            indP = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
        else:
            indP = pre * 1.8e-9 * H**2 * np.exp(4*np.pi*xi)/xi**6
            indM = pre * 8.6e-7 * H**2 * np.exp(4*np.pi*xi)/xi**6
        
        #Factors of two to match my convention
        PTanalytic = {"tot":(2*pre + indP + indM), "vac":2*pre, "ind+":2*indP, "ind-":2*indM}
        k = H*np.exp(self._N)
        return k, PTanalytic
    


