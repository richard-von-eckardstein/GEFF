import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import solve_ivp, trapezoid

from geff._docs.docs_pt import DOCS
from geff.mbm import GaugeSpec
from geff.bgtypes import BGSystem, Constant

from typing import Tuple
from types import NoneType

__doc__ = DOCS["module"]

class PowSpecT:
    r"""
    A class used to compute the tensor power spectrum including vacuum and gauge-field induced contributions.

    These main method of this module is `compute_pt`, which computes both the vacuum and gauge-field induced contribution
      to the tensor power spectrum, $\mathcal{P}_{T,\lambda}^{\mathrm{vac}}$ and $\mathcal{P}_{T,\lambda}^{\mathrm{ind}}$.

    Results are internally computed using numerical units, but are returned in physical units.
    """
    def __init__(self, insys : BGSystem):
        """
        Initialise the class from a GEF solution.

        Parameters
        ----------
        insys : BGSystem
            the GEF solution.
        """
        #Set GEF results to Hubble units.
        sys = BGSystem.from_system(insys, True)
        sys.units = False
        
        #import the background evolution
        N = sys.N.value
        a = np.exp(N)
        H = sys.H.value

        self._omega = sys.omega
            
        #Set the range of modes
        self.maxk = max(a*H)
        self.mink = 1e4

        #Define Useful quantities

        self._t = sys.t.value
        self._N = N
        self._H = H
        self._xi= sys.xi.value

        #define interpolated quantities as needed
        self._af = CubicSpline(self._t, a)
        if isinstance(self._H, Constant):
            self._Hf = lambda t: self._H
        else:
            self._Hf = CubicSpline(self._t, H)
        self._HN = CubicSpline(self._N, self._H)
        self._khN = CubicSpline(N, sys.kh)

        #Obtain eta as a functio of time for phases
        def deta(t, y): return 1/self._af(t)
        
        soleta = solve_ivp(deta, [min(self._t), max(self._t)], np.array([0]), t_eval=self._t)

        self._etaf = CubicSpline(self._t, soleta.y[0,:])
        return
    
    def compute_pt(self, nmodes : int, spec : GaugeSpec, FastGW : bool=True,
                    atols : list=[1e-3,1e-20], rtols : list=[1e-4,1e-4], momgrid : int=100
                    ) -> Tuple[np.ndarray,dict]:
        r"""
        Compute the full tensor power spectrum.

        The method lodes data on gauge modes $A_\lambda(t,k)$ from a file indicated by `mbm_file`.
        The power spectrum is evaluated fpr `nmodes` log-spaced momenta $k \in [10^{4}a(0)H(0), 10 a(t_{\rm max}) H(t_{\rm max})]$

        Parameters
        ----------
        nmodes : NDArray
            the number of momenta
        ModePath : str
            path to a file containing gauge-field mode functions
        FastGW : bool
            If `True`, only the expected dominant contributions to $\mathcal{P}_{T,\lambda}^{\mathrm{ind}}$ are computed
        atols : list
            absolute tolerance for `compute_homogeneous` (index 0) and `compute_green` (index 1)
        rtols : list
            relative tolerance for `compute_homogeneous` (index 0) and `compute_green` (index 1)
        momgrid : int
            passed to `compute_ptind`
        
        Returns
        -------
        ks : NDArray
            the momenta $k$
        PT : dict
            a dictionary containing all contributions to the tensor power spectrum and its total.
        """

        k = np.logspace(np.log10(self.mink), np.log10(10*self.maxk), nmodes)
        ks, tstarts = self._find_tinit_bd(k, mode="k")
        
        Ngrid = spec["N"]
        tgrid = spec["t"]
        pgrid = spec["k"]

        GaugeModes = {"+":(spec["Ap"], spec["dAp"]), "-":(spec["Am"], spec["dAm"])}
        
        indend = np.argmin(abs(Ngrid-max(self._N)))

        PT = {"tot":[], "vac":[], "ind+,++":[], "ind+,+-":[], "ind+,--":[], "ind-,++":[], "ind-,+-":[], "ind-,--":[]}

        GWpols = [("+", 1), ("-",-1)]

        gaugepols=[(("+",1),("+",1)),
                    (("+",1),("-",-1)),
                        (("-",-1),("-",-1))]

        sign = np.sign(self._xi[0])

        for i, k in enumerate(ks):
            tstart = tstarts[i]

            if k > 10**(5/2)*self.maxk:
                for key in PT.keys():
                    PT[key].append(0)
            else:
                uk, _ = self.compute_homogeneous(k, tstart, tgrid, atol=atols[0], rtol=rtols[0])
                Green = self.compute_green(k, uk, tstart, tgrid, indend, atol=atols[1], rtol=rtols[1])
                
                PT["vac"].append(self.compute_ptvac(k, uk[indend]))

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
                                self.compute_ptind(k, GWpol, Ngrid, indend, Green, pgrid, lx, Ax, dAx, ly, Ay, dAy, momgrid) )

        PT["tot"] = np.zeros(ks.shape)
        for key in PT.keys():
            PT[key] = np.array(PT[key])
            if ("+" in key) or ("-" in key):
                PT["tot"] += 0.5*PT[key]
            elif key=="vac":
                PT["tot"] += PT[key]

        #this returns k with units restored. Needs to be matched by omega_GW
        return ks*self._omega, PT
    
    def compute_homogeneous(self, k : float, tvac : float, teval : np.ndarray|NoneType, atol : float=1e-3, rtol : float=1e-4) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the evolution of a vacuum mode starting from Bunch&ndash;Davies vacuum.

        For a mode $u_0(t,k)$ with momentum $k$, initialise in Bunch&ndash;Davies when $k = 10^{5/2} a(t_{\rm vac})H(t_{\rm vac})$.
        Its evolution is obtained by solving `tensor_mode_eq` using `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        k : float
           the momentum $k$.
        tvac : float
            the initial time, $t_{\rm init}$
        teval : NDArray or None
            time coordinates $t$ of $u_0(t,k)$ (`None`: same as class input `sys`) 
        atol : float
            the absolute tolerance for `solve_ivp`
        rtol : float
            the relative tolerance for `solve_ivp`

        Returns
        -------
        u : NDArray
            the vacuum tensor modes, $\sqrt{2k}u(t, k)$
        duk : NDArray
            the derivative of the vacuum tensor modes, $a\sqrt{2/k}\dot{u}(t, k)$
        """

        if len(teval)==0:
            teval = self._t
        tend = max(teval)

        #conformal time needed for relative phases
        eta = self._etaf(teval)

        istart = 0
        while teval[istart]<tvac:
            istart+=1
        
        #define the ODE for the GW modes
        def ode(t, y): return self.tensor_mode_eq( y, k, self._Hf(t), self._af(t) )
        
        #Initialise the modes in Bunch Davies
        Zini = np.array([1, -10**(-5/2), 0, -1])

        sol = solve_ivp(ode, [tvac, tend], Zini, t_eval=teval[istart:], method="RK45", atol=atol, rtol=rtol)
        if not(sol.success):
            print("Something went wrong")

        #the mode was in vacuum before tvac
        vac = list( np.exp(-1j*eta[:istart]*k) )
        dvac = list( -1j*np.exp(-1j*eta[:istart]*k) )

        #Create an array tracking a modes evolution from Bunch Davies to late times. Ensure equal length arrays for every mode k
        uk = np.array( vac + list( (sol.y[0,:] + 1j*sol.y[2,:])*np.exp(-1j*k*eta[istart]) ) )/self._af(teval)
        duk = np.array( dvac + list( (sol.y[1,:] + 1j*sol.y[3,:])*np.exp(-1j*k*eta[istart]) ) )/self._af(teval)

        return uk, duk
    
    @staticmethod
    def tensor_mode_eq(y : np.ndarray, k : float, H : float, a : float) -> np.ndarray:
        r"""
        Mode equation for vacuum tensor modes.

        Parameters
        ----------
        y : NDArray
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
        dydt : NDArray
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
    
    def compute_green(self, k : float, uk : np.ndarray, tvac : float, teval : np.ndarray|NoneType, ind : int, atol : float=1e-30, rtol : float=1e-4) -> np.ndarray:
        r"""
        Compute the evolution of the Green function $G_k(t',t)$ for fixed $t'$.

        The evolution is obtained by solving `green_ode` from $t=t'$ backwards until $t_{\rm vac}$, defined through $k = 10^{5/2} a(t_{\rm end})H(t_{\rm vac})$.
        From $t_{\rm end}$ onwards, the Green function is instead computed from the mode $u_0(t, k)$. The method uses `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        k : float
           the momentum $k$.
        uk : NDArray
            the mode function $u_0(t,k)$
        tvac : float
            the final time, $t_{\rm vac}$
        teval : NDArray or None
            time coordinates $t$ of $G_k(t',t)$ (`None`: same as class input `sys`) 
        ind : int
            index of teval corresponding to $t'$
        atol : float
            the absolute tolerance for `solve_ivp`
        rtol : float
            the relative tolerance for `solve_ivp`

        Return
        ------
        GreenN : NDArray
            the Green function $-k G_k(t', t)$ 
        """
        if len(teval)==0:
            teval = self._t
            
        istart = 0
        while teval[istart]<tvac:
            istart+=1

        # G(k, t, t) = 0 by definition
        Aini = np.array([0, 1])
    

        # Solve the EoM for G backwards in time starting from G(k, t, t)
        def Aode(t, y): return -self.green_ode(y, k, self._Hf(-t), self._af(-t))
        solA = solve_ivp(Aode, [-teval[ind], -tvac], Aini, t_eval=-teval[istart:ind+1][::-1],
                         method="RK45", atol=atol, rtol=rtol)

        #For numerical stability, only solve the EoM for the Green function until t' = tvac. Afterwards, compute it directly from the vacuum modes.
        GreenN = np.zeros(teval.shape)
        GreenN[istart:ind+1] = solA.y[0,:][::-1]
        GreenN[:istart] = ( (uk[ind].conjugate()*uk).imag*self._af(teval)**2 )[:istart]

        return GreenN
    
    @staticmethod
    def green_ode(y : np.ndarray, k : float, H : float, a : float) -> np.ndarray:
        r"""
        Evolve the re-scaled Green function for tensor modes in time.

        The re-scaled Green function is
        $$B_{t'}^k(t) = 2 k a(t)^2 \operatorname{Im} \left[ u_0^*(t', k) \, u_0(t,k) \right] = k G_k(t', t) \,.$$
        
        whose ODE is coupled to 

        $$C_{t'}^k(t) = 2 a(t)^3 \operatorname{Im} \left[ u_0^*(t', k) \, {\dot{u}_0}(t,k) \right] \,.$$

        Parameters
        ----------
        y : NDArray
            the functions B and C
        k : float
            comoving momentum $k$
        H : float
            Hubble rate, $H(t)$
        a : float
            scale factor, $a(t)$

        Returns
        -------
        dydt : NDArray
            an array of time derivatives of A

        """
        dydt = np.zeros(y.shape)
        dydt[0] =(2*H*y[0] + k/a*y[1])
        dydt[1] = -k/a*y[0]
        return dydt

    
    def _find_tinit_bd(self, init : np.ndarray, mode : str="k") -> Tuple[np.ndarray, np.ndarray]:
        """
        Determines the pair of $k$ and $t$ satisfying $k = 10^(5/2)a(t)H(t)$.

        Depending on `mode`, `init` may be a time coordinate (`mode='t'`), $e$-folds (`mode='N'`) or momentum (`mode='k'`).

        Parameters
        ----------
        init : array
            the input array (t, N, or k)
        mode : str
            indicate the type of `init`

        Returns
        -------
        ks : NDarray
            an array of momenta
        tstart : NDarray
            an array of times

        Raises
        ------
        KeyError
            if `mode` is not 't, 'k' or 'N'
        """

        pwr = 5/2

        t = self._t
        def logkH(t): return np.log(self._af(t)*self._Hf(t))
        if mode=="t":
            tstart = init
            logks = logkH(tstart)
            ks = 10**(pwr)*np.exp(logks)

        elif mode=="k":
            ks = init

            tstart = []
            for k in ks:
                ttmp  = self._t[np.searchsorted(10**(pwr)*np.exp(logkH(self._t)), k, "right")]
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(self._N, t)(init)
            ks = 10**pwr*np.exp(logkH(tstart))

        else:
            raise KeyError("'mode' must be 't', 'k' or 'N'")

        return ks, tstart
        
    def compute_ptvac(self, k : np.ndarray, uk : np.ndarray) -> np.ndarray:
        r"""
        Compute the vacuum power spectrum, $\mathcal{P}_{T,\lambda}^{\mathrm{vac}}(t,k)$.

        Parameters
        ----------
        k : NDArray
            momentum, $k$
        uk : NDArray
            vacuum tensor mode $u_0(t,k)$

        Return
        ------
        PTvac : NDArray
            the vacuum tensor power spectrum
        """
        PTvac = 2*(k*self._omega)**2/( np.pi**2 ) * abs(uk)**2
        return PTvac

    def compute_ptind(self, k : float, lgrav : float, Ngrid : np.ndarray, ind: int, GreenN : np.ndarray, pgrid : np.ndarray,
                                l1 : float, A1 : np.ndarray, dA1 : np.ndarray, l2 : float, A2 : np.ndarray, dA2 : np.ndarray,
                                momgrid : int=100) -> float:
        r"""
        Compute the vacuum power spectrum, $\mathcal{P}_{T,\lambda}^{\mathrm{vac}}(t,k)$.

        The integral is computed by integrating over the inner integral over the gauge-mode functions $A_\mu(s,p)$ using `scipy.integrate.trapezoid`.
        The external momentum integrals are also computed using `scipy.integrate.trapezoid` on a grid of momenta momgrid x momgrid. The mode functions
        are interpolated over momentum to match this grid.

        Parameters
        ----------
        k : float
            momentum $k$
        lgrav : NDArray
            helicity $\lambda$
        Ngrid : NDArray
            $e$-folds $\log a$
        ind : integer
            the power spectrum is computed at $t$ corresponding to `Ngrid[ind]`
        GreenN : NDArray
            the Green function $kG_k(t, s)$ with $s$ corresponding to `Ngrid`
        pgrid : NDArray
            momenta $p$ for the mode functions $A_\mu(s,p)$
        l1, l2 : float
            the gauge-field helicities $\mu_1$, $\mu_2$
        A1, A2 : float
            the gauge-field mode functions $\sqrt{2k} A_{\mu_1, \mu_2}(s, p)$
        dA1, dA2 : float
            the gauge-field mode function derivatives $a(s)\sqrt{2/k} \dot{A}_{\mu_1, \mu_2}(s, p)$
        momgrid : int
            the grid size for the momentum integrals.

        Returns
        -------
        PTind : float
            the gauge-field induced tensor power spectrum
        """

        cutUV = self._khN(Ngrid[ind])/k
        cutIR = min(pgrid)/k
        HN = self._HN(Ngrid)

        logAs = np.linspace(np.log(max(0.5, cutIR)), np.log(cutUV), momgrid)

        Ax_rad = PchipInterpolator(np.log(pgrid), abs(A1))
        Ax_phase = PchipInterpolator(np.log(pgrid), np.arccos(A1.real/abs(A1)))
        
        dAx_rad = PchipInterpolator(np.log(pgrid), abs(dA1))
        dAx_phase = PchipInterpolator(np.log(pgrid), np.arccos(dA1.real/abs(dA1)))

        Ay_rad = PchipInterpolator(np.log(pgrid), abs(A2))
        Ay_phase = PchipInterpolator(np.log(pgrid), np.arccos(A2.real/abs(A2)))
        
        dAy_rad = PchipInterpolator(np.log(pgrid), abs(dA2))
        dAy_phase = PchipInterpolator(np.log(pgrid), np.arccos(dA2.real/abs(dA2)))

        def Afuncx(x): return Ax_rad(x)*np.exp(1j*Ax_phase(x))
        def dAfuncx(x): return dAx_rad(x)*np.exp(1j*dAx_phase(x))

        def Afuncy(x): return Ay_rad(x)*np.exp(1j*Ay_phase(x))
        def dAfuncy(x): return dAy_rad(x)*np.exp(1j*dAy_phase(x))

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
                mask = np.where(z<self._khN(Ngrid)/k, 1, 0)

                val = (dAx*dAy + l1*l2*Ax*Ay)*mask*k/(np.exp(3*Ngrid)*HN)
                timeintegrand = GreenN*val*mom
            
                timeintre = trapezoid(timeintegrand[:ind].real, Ngrid[:ind])
                
                timeintim = trapezoid(timeintegrand[:ind].imag, Ngrid[:ind])
                
                IntInner[j] = (timeintre**2 + timeintim**2)*A

  
            IntOuter.append(trapezoid(IntInner, Bs))
        IntOuter = np.array(IntOuter)

        PTind = trapezoid(IntOuter, logAs) / (16*np.pi**4)*(k*self._omega)**4

        return PTind
    


