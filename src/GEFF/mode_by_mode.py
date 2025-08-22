import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.integrate import quad, simpson

from GEFF.bgtypes import Val, BGSystem
from GEFF.utility.mbm_funcs import mode_equation_classic, bd_classic

from typing import Tuple, Callable
from types import NoneType

class GaugeSpec(dict):
    r"""
    A class representing a spectrum of gauge-field modes as a function of time.

    This class inherits from `dict` and needs the following keys:  
    't', 'N', 'k', 'Ap', 'dAp', 'Am', 'dAm'

    The spectrum can be evaluated at certain times $t$ or for certain momenta $k$ by using `tslice` and `kslice`
    Furthermore, the spectrum contained in the object can be integrated to compute gauge-field expectation values.
     The result can be used to estimate the error of a GEF run.

    Attributes
    ----------
    t : NDArray
        the cosmic time coordinates $t$ of the spectrum
    N : NDArray
        the $e$-folds as a function of cosmic time, $N(t)$
    k : NDarray
        the momenta $k$ at which the spectrum is evaluated
    Ap, Am : NDarray
        the mode functions, $\sqrt{2 k} A_\pm(k, t)$
    dAp, dAm : NDarray
        the mode-function derivatives, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
    """

    def __init__(self, in_dict : dict):
        """
        Initialise the spectrum from a dictionary.

        Parameters
        ----------
        in_dict : dict
            dictionary with keys 't', 'N', 'k', 'Ap', 'dAp', 'Am', 'dAm'

        Raises
        ------
        KeyError
            if a key in {'t', 'N', 'k', 'Ap', 'dAp', 'Am', 'dAm'} is missing.
        ValueError
            if `len(in_dict['t'])` does not match `len(in_dict['N'])` or if
            `in_dict['X']).shape` does not match `(len(in_dict['k']),len(in_dict['t']))` for 'X' in {'Ap', 'dAp', 'Am', 'dAm'}.
        """
        for key in ["t", "N", "k", "Ap", "dAp", "Am", "dAm"]:
            if key not in in_dict.keys():
                raise KeyError(f"Missing key: {key}")
            
        if not(len(in_dict["t"]) == len(in_dict["N"])):
            raise ValueError("The length of 't' needs to match the length of 'N'")
        
        shape = (len(in_dict["k"]), len(in_dict["t"]))
        for key in ["Ap", "dAp", "Am", "dAm"]:
            if not( in_dict[key].shape == shape):
                raise ValueError(f"The shape of {key} needs to be {shape}") 
        
        super().__init__(in_dict)

    @classmethod
    def read_spec(cls, path : str):
        """
        Initialise the class from a file.

        Parameters
        ----------
        path : str
            path to the data 

        Returns
        -------
        spec : GaugeSpec
            the imported spectrum
        """
        input_df = pd.read_table(path, sep=",")
        dataAp = input_df.values

        x = np.arange(3,dataAp.shape[1], 4)
        
        t = np.asarray(dataAp[1:,1], dtype=float)
        N = np.asarray(dataAp[1:,2], dtype=float)
        logk = np.array([(complex(dataAp[0,y])).real for y in x])
        Ap = np.array([[complex(dataAp[i+1,y]) for i in range(len(N))] for y in x])
        dAp = np.array([[complex(dataAp[i+1,y+1]) for i in range(len(N))] for y in x])
        Am = np.array([[complex(dataAp[i+1,y+2]) for i in range(len(N))] for y in x])
        dAm = np.array([[complex(dataAp[i+1,y+3]) for i in range(len(N))] for y in x])

        k = 10**logk

        spec = cls({"t":t, "N":N, "k":k,
                        "Ap":Ap, "dAp":dAp, "Am":Am, "dAm":dAm})
        
        return spec
    
    def save_spec(self, path : str):
        """
        Store the spectrum in a file.

        Parameters
        ----------
        path : str
            path to the data 
        """
        thinning = 1
        N = np.array([np.nan]+list(self["N"][-1::-thinning][::-1]))
        t = np.array([np.nan]+list(self["t"][-1::-thinning][::-1]))

        dic = {"t":t, "N":N}

        for j, k in enumerate(self["k"]):
            spec_slice = self.kslice(j)
            logk = np.log10(spec_slice["k"])
            for key in ["Ap", "dAp", "Am", "dAm"]:
                dictmp = {key + "_" + str(j) :np.array([logk] + 
                                                       list(spec_slice[key][-1::-thinning][::-1]))}
                dic.update(dictmp)
    
        dir_name = os.getcwd()
    
        path = os.path.join(dir_name, path)
    
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)
        
        return

    def get_dim(self) -> dict:
        """
        Get the number of time coordinates and momenta stored in the spectrum.

        Returns
        -------
        dict
            a dictionary encoding the spectrum's shape
        """
        return {"kdim":len(self["k"]), "tdim":len(self["t"])}
    
    def tslice(self, ind : int) -> dict:
        """
        Evaluate the spectrum at time `self['t'][ind]`.

        Parameters
        ----------
        ind : int
            the temporal index

        Returns
        -------
        spec_slice : SpecSlice
            the spectrum at time `self['t'][ind]`
        """

        spec_slice = {}
        for key in self.keys():
            if key in ["N", "t", "cut"]:
                spec_slice[key] = self[key][ind]
            elif key=="k":
                spec_slice[key] = self[key]
            else:
                spec_slice[key] = self[key][:,ind]
        return SpecSlice(spec_slice)
    
    def kslice(self, ind : int) -> dict:
        """
        Obtain the time evolution for the momentum `self['k'][ind]`.

        Parameters
        ----------
        ind : int
            the momentum index.

        Returns
        -------
        spec_slice : dict
            a dictionary with keys like `self`  
        """

        spec_slice = {}
        for key in self.keys():
            if key in ["N", "t", "cut"]:
                spec_slice[key] = self[key]
            elif key=="k":
                spec_slice[key] = self[key][ind]
            else:
                spec_slice[key] = self[key][ind,:]
        return spec_slice
    
    def integrate(self, BG : BGSystem, n : int=0, cutoff="kh", **IntegratorKwargs) -> np.ndarray:
        r"""
        Compute the three integrals

        $$ \mathcal{F}_\mathcal{E}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm h}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\,,$$
        $$ \mathcal{F}_\mathcal{G}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{a k^{n+4}}{2 \pi^2 k_{{\rm h}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)]\,,$$
        $$ \mathcal{F}_\mathcal{B}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm h}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2\,,$$

        for a given $n$ and each time coordinate $t$ in the spectrum.

        If the time coordinates stored in `BG` do not match those stored in the spectrum, $k_{\rm h}(t)$ is evaluated using interpolation.

        Parameters
        ----------
        BG : BGSystem
            a system containing the UV cut-off, $k_{\rm h}(t)$
        n : int
            the integer $n$ in $\mathcal{F}_\mathcal{X}^{(n)}(t)$ for $\mathcal{X} = \mathcal{E}, \mathcal{B},\mathcal{G}$
        cutoff : str
            the name under which the UV-cutoff is stored in `BG`
        **IntegratorKwargs :  kwargs
            passed to `SpecSlice.integrate_slice`
        

        Returns
        -------
        FMbM : NDArray
            $\mathcal{F}_\mathcal{E}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$ stored in a shape (N, 3, 2).
            The first index corresponds to time $t$, the second index to $\mathcal{X}$, the third index is the integral result (at 0) and its error (at 1).
        """

        self._add_cutoff(BG, cutoff)

        tdim = self.get_dim()["tdim"]

        FMbM = np.zeros((tdim, 3,2))
        for i in range(tdim):
            spec_slice = self.tslice(i)
            FMbM[i,:] = spec_slice.integrate_slice(n=n, **IntegratorKwargs)
        
        return FMbM
    
    def estimate_GEF_error(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                                    err_thr : float=0.025, binning : int|NoneType=5, verbose : bool=True,
                                    **IntegratorKwargs) -> Tuple[list, np.ndarray, list]:
        r"""
        Estimate the relative deviation between a GEF solution and the mode spectrum by computing

        $$\varepsilon_\mathcal{X} = \left|1 - \frac{\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}}{\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}}\right|$$

        for $\mathcal{X} = \mathcal{E},\,\mathcal{B},\,\mathcal{G}$. Here, $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$ are the integrals computed by `integrate`, $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ refer
          to the same quantity stored in `BG`.
        If the time coordinate of `BG` does not align with the spectrum, its values are interpolated.

        Because $k_{\rm h}(t)$ increases monotonically, the spectrum contains only few relevant modes $k < k_{\rm h}(t)$ at early times.
        This poses a problem for the numerical integration of $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$.
        To avoid claiming a disagreement between $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$ and $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ due to this effect,
        errors with $\varepsilon_\mathcal{X} > \varepsilon_{\rm thr}$ are discarded until the first time when $\varepsilon_\mathcal{X} < \varepsilon_{\rm thr}$.

        As the integration result fluctuates significantly for few momenta $k < k_{\rm h}(t)$ when using `simpson`,
        the errors can be binned by setting `binning`. The reported error is the average over a bin of width $(t_{i}, t_{i+\Delta})$ with $\Delta$ set by `binning`.
        This binned error is then associated to the time $(t_{i} + t_{i+\Delta})/2$. For `quad`, `binning` can also be set to `None`.
        For details on the integration methods `simpson` and `quad`, see `SpecSlice.integrate_slice`.

        Parameters
        ----------
        BG : BGSystem
            the system where $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ is stored
        references : list of str
            the names where $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ are stored in `BG`
        cutoff : str
            the name where the UV-cutoff, $k_{\rm h}$, is stored in `BG`
        err_thr : float
            the error threshold $\varepsilon_{\rm thr}$
        binning : int or None
            the bin size $\Delta$ (no binning if `None`)
        verbose : bool
            if `True`, print a summary of the errors
        **IntegratorKwargs :  kwargs
            passed to `SpecSlice.integrate_slice`

        Returns
        -------
        errs : list of NDArray
            a list of the binned errors with entries $[\varepsilon_\mathcal{E},\varepsilon_\mathcal{B}, \varepsilon_\mathcal{G}]$
        terr : NDArray
            the time coordinates corresponding to `errs`
        og_errs : list of NDArray
            the same as `errs` but without binning
        """

        og_errs = self._estimate_error(BG, references, cutoff, **IntegratorKwargs)
        terr = self["t"]

        if binning is not None:
            errs, terr = self._bin_error(og_errs, binning)
        else:
            errs = og_errs
        
        errs, terr = self._process_error(errs, terr, err_thr)

        if verbose:
            self._error_summary(errs, terr, references)

        return errs, terr, og_errs
    
    def merge_spectra(self, spec : 'GaugeSpec'):
        """
        Combine two spectra with the same momenta $k$ but unequal times $t$.

        Parameters
        ----------
        spec : GaugeSpec
            the second spectrum

        Raises
        ------
        AssertionError
            if the momenta $k$ do not match up.
        
        """
        assert (spec["k"] == self["k"]).all()

        ind = np.where(self["t"]<=spec["t"][0])[0][-1]

        if "cut" in self.keys():
            self.pop("cut")

        for key in self.keys():
            if key in ["t", "N"]:
                self[key] = np.concatenate([self[key][:ind], spec[key]])
            else:
                if key != "k":
                    self[key] = np.concatenate([self[key][:,:ind], spec[key]], axis=1)
        return
    
    def add_momenta(self, spec : 'GaugeSpec'):
        """
        Combine two spectra with the same times $t$ but unequal momenta $k$.

        Parameters
        ----------
        spec : GaugeSpec
            the second spectrum

        Raises
        ------
        AssertionError
            if the times $t$ do not match up.
        
        """

        assert (np.round(spec["t"],1) == np.round(self["t"],1)).all()

        newks = []
        mds = ["Ap", "dAp", "Am", "dAm"]
        newmodes = dict( zip(mds, [ [] for i in mds]) )

        for i, k in enumerate(self["k"]):
            mask = np.where(k > spec["k"])[0]
            if len(mask) != 0:
                newks.append(spec["k"][mask])    
                spec["k"] = np.delete(spec["k"], mask)
                for md in mds:
                    newmodes[md].append(spec[md][mask,:])
                    spec[md] = np.delete(spec[md], mask, axis=0)
            newks.append(np.array([k]))
            for md in mds:
                newmodes[md].append(np.array([self[md][i,:]]))
        
        mask = np.where(spec["k"] > k)[0]
        if len(mask) != 0:
            newks.append(spec["k"][mask])    
            spec["k"] = np.delete(spec["k"], mask)
            for md in mds:
                newmodes[md].append(spec[md][mask,:])
                spec[md] = np.delete(spec[md], mask, axis=0)

        self["k"] = np.concatenate(newks)
        for md in mds:
            self[md] = np.concatenate(newmodes[md], axis=0)

        return
    
    def remove_momenta(self, ind):
        """
        Remove the spectrum at momentum `self["k"][ind]`.

        Parameters
        ----------
        ind : int
            the index at which to remove the spectrum entry
        """
        self["k"] = np.delete(self["k"], ind)
        for md in ["Ap", "dAp", "Am", "dAm"]:
            self[md] = np.delete(self[md], ind, axis=0)
        return
    
    def _check_overlap(self, t):
        mask = np.isin(t, self["t"], assume_unique=True)
        if len(t[mask]) != len(self["t"]):
            #this should be a warning
            print("The times in the current GaugeSpec instance are " \
            "not a subset of the times in the BGSystem. Reverting to interpolation.")

            return False, None
        else:
            return True, mask
    
    def _add_cutoff(self, BG : BGSystem, cutoff="kh"):
        units = BG.get_units()
        BG.set_units(False)

        scale = getattr(BG, cutoff)

        bl, mask = self._check_overlap(BG.t)

        if bl:
            self["cut"] = scale[mask]
        else:
            self["cut"] = CubicSpline(BG.t, scale)(self["t"])
        
        BG.set_units(units)

        return self["cut"]
    
    def _get_reference(self, BG : BGSystem, references=["E", "B", "G"], cutoff="kh"): 
        units = BG.get_units()
        BG.set_units(False)

        scale = getattr(BG, cutoff)

        bl, mask = self._check_overlap(BG.t)

        Fref = []
        for val in references: 
            val_arr = (getattr(BG, val)*(BG.a/scale)**4)
            if bl:
                Fref.append( val_arr[mask] )
            else:
                Fref.append( CubicSpline(BG.t, val_arr)(self["t"]) ) 

        BG.set_units(units)

        return Fref
    
    
    def _estimate_error(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                         **IntegratorKwargs):
        FMbM = self.integrate(BG, n=0, cutoff=cutoff, **IntegratorKwargs)
        Fref = self._get_reference(BG, references, cutoff)

        errs = []

        for i, spl in enumerate(Fref):
            err =  np.minimum(1e3, abs( 1 - FMbM[:,i,0]/ spl ))
            errs.append(np.where(np.isnan(err), 10.0, err))

        return errs
    
    def _process_error(self, errs, terr, err_thr):
        removals = []
        for err in errs:
            #remove the first few errors where the density of modes is low:
            removals.append(np.where(err < err_thr)[0][0])
        #ind = 0
        ind = max(removals)
        errs = [err[ind:] for err in errs]
        terr = terr[ind:]

        return errs, terr
    
    def _bin_error(self, errs, binning):
        terr = self["t"]

        bin_terr = terr[::-binning][::-1]
        tbins = bin_terr[1:] + (bin_terr[1:] - bin_terr[:-1])/2
        count, _  = np.histogram(terr, bins=tbins)

        bin_errs = []
        for err in errs:
            sum, _  = np.histogram(terr, bins=tbins, weights=err)
            bin_errs.append(sum/count)
        bin_terr = bin_terr[2:] 

        return bin_errs, bin_terr
    
    @staticmethod
    def _error_summary(bin_errs, bin_terr, references : list[str]=["E", "B", "G"]):
        print("The mode-by-mode comparison finds the following relative deviations from the GEF solution:")
        for i, key in enumerate(references):
            err = bin_errs[i]
            errind = np.where(err == max(err))
            rmserr = np.round(100*np.sqrt(np.sum(err**2)/len(err)), 1)
            maxerr = np.round(100*err[errind][0], 1)
            tmaxerr = bin_terr[errind][0]#np.round(Nerr[errind][0], 1)
            errend = np.round(100*err[-1], 1)
            terrend = bin_terr[-1]#np.round(Nerr[-1], 1)
            print(f"\t-- {key} --")
            print(f"max: {maxerr}% at t={tmaxerr}")
            print(f"final: {errend}% at t={terrend}")
            print(f"RMS: {rmserr}%")
        return

   
# continue from here next time
class SpecSlice(dict):
    r"""
    A class representing a spectrum of gauge-field modes at a time $t$.

    Instances of this class are created by `GaugeSpec.tslice`. The main purpose of this class is to integrate the spectrum at time $t$ using `integrate_slice`.

    Attributes
    ----------
    t : NDArray
        the cosmic time coordinates $t$ of the spectrum
    N : NDArray
        the $e$-folds as a function of cosmic time, $N(t)$
    k : NDarray
        the momenta $k$ at which the spectrum is evaluated
    Ap, Am : NDarray
        the mode functions, $\sqrt{2 k} A_\pm(k, t)$
    dAp, dAm : NDarray
        the mode-function derivatives, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
    """

    def __init__(self, in_dict):
        super().__init__(in_dict)

    def integrate_slice(self, n : int=0, method="simpson", modethr : int=100, epsabs : float=1e-20, epsrel : float=1e-4, interpolator=PchipInterpolator) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the three integrals

        $$ \mathcal{F}_\mathcal{E}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm h}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2,$$
        $$ \mathcal{F}_\mathcal{G}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{a k^{n+4}}{2 \pi^2 k_{{\rm h}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)]$$
        $$ \mathcal{F}_\mathcal{B}^{(n)}(t) = \int\limits_{0}^{k_{{\rm h}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm h}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2$$

        for a fixed time $t$ and index $n$.

        The integrals can either be computed directly using `simpson` or `quad` from `scipy.interpolate`. When using `quad` the data for $\sqrt{2 k} A_\pm(k, t)$, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
          are interpolated to obtain smooth functions. To avoid this, it is recommended to use `simpson`.

        When using `simpson`, the integral is only computed if $m > m_{\rm thr}$ momenta $k_i$ satisfy $k < k_{\rm h}$. Otherwise, the integral is set to zero.

        When using `quad`, the absolute and relative tolerances of the integrator are set by `epsabs` and `epsrel`. The interpolation method is defined by `interpolator`.
        Currently, only `CubicSpline` and `PchipInterpolator` from `scipy.interpolate` are supported. The later is preferred as interpolating the oscillatory mode functions can be subject to "overshooting".
        See [scipy's tutorial for 1-D interpolation](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection) for more details.

        Parameters
        ----------
        n : int
            the integer $n$ in $\mathcal{F}_\mathcal{X}^{(n)}(t)$ for $\mathcal{X} = \mathcal{E}, \mathcal{B},\mathcal{G}$
        method : str
            set the integration method to `simpson` or `quad`
        modethr : int
            set $m_{\rm thr}$ when using `simpson`
        epsabs : float
            the absolute tolerance of `quad`
        epsrel : float
            the relative tolerance of `quad`  
        interpolator
            the interpolator used to get smooth functions for `quad`
        

        Returns
        -------
        FMbM : NDArray
            contains [$\mathcal{F}_\mathcal{E}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$] and the error estimated by `quad`.
             When using `simpson` the error is a dummy output. The shape of the result is (3, 2) with the second index indicating the integral (at 0), or the error (at 1).
        """
        x = (n+4)*np.log(self["k"]/self["cut"])

        helicities = ["p", "m"]
        integs = np.zeros((3, 2, len(x)))
        for i, lam in enumerate(helicities):
            sgn = np.sign(0.5-i)
            integs[0,i,:] = self._Espec(lam)
            integs[1,i,:] = self._Bspec(lam)
            integs[2,i,:] = sgn*self._Gspec(lam)

        res = np.zeros((3,2))
        
        for i in range(3):
            if method=="simpson":
                msk = np.where(x < 0)[0]
                if len(msk) < modethr: #cannot trust simpsons integration for too few modes.
                    return res
                x = x[msk]
                res[i,0] = (self._simpson_integrate( integs[i,0,msk] ,x) 
                                         + (-1)**n*self._simpson_integrate(integs[i,1,msk], x) )
                res[i,1] = 1e-6*res[i,0]

            elif method=="quad":
                resp = self._quad_integrate( integs[i,0,:], x, epsabs, epsrel, interpolator)
                resm = self._quad_integrate( (-1)**n*integs[i,1,:], x, epsabs, epsrel, interpolator)
                res[i,:] = resp +resm
            
        res = 1/(2*np.pi)**2*res/(n+4)

        res[:,1] = abs(res[:,1]/res[:,0])
        return res

    def _Espec(self, lam):
        return abs(self["dA"+lam])**2
    
    def _Bspec(self, lam):
        return abs(self["A"+lam])**2
    
    def _Gspec(self, lam):
        return (self["A"+lam].conjugate()*self["dA"+lam]).real
    
    def _simpson_integrate(self, integrand, x):
        integrand = integrand*np.exp(x)
        return simpson(integrand, x)

    def _quad_integrate(self, integrand, x, epsabs : float=1e-20, epsrel : float=1e-4, interpolator=PchipInterpolator):
        msk = np.where(abs(integrand) > 1e-1*max(epsrel*max(abs(integrand)), epsabs))[0]
        if len(msk) > 0:
            spl = interpolator(x, np.arcsinh(integrand))
            def f(x): return np.sinh(spl(x))*np.exp(x)
            val, err = quad(f, x[msk][0], 0., epsabs=epsabs, epsrel=epsrel)
            return np.array([val, err])
        else:
            return np.nan*np.ones((2))

class BaseModeSolver:
    r"""
    A class used to compute gauge-field modes evolving on a time-dependent background.

    This class is used to evolve the gauge-field modes $A_\pm(t,k)$ and their derivatives in time by using the
    evolution of the time-dependent background obtained from a GEF solution.
    
    The evolution is determined by an ODE for the four (complex) variables 

    $$\sqrt{2k} A_\lambda(t,k), \quad a(t) \sqrt{\frac{2}{k}}\dot{A}_\lambda(k, t), \quad \lambda = \pm 1$$

    in terms of their real and imaginary parts. By default, the evolution equation is

    $$ \ddot{A}_\lambda(t,k) + H \dot{A}_\lambda(t,k) +  \left[ \left(\frac{k}{a}\right)^2  - 2\lambda \left(\frac{k}{a}\right) \xi H\right] A_\lambda(t,k) = 0$$
    
    with the evolution for $H(t)$, $a(t)$, $\xi(t)$ obtained from the GEF solution.

    The modes are initialized deep inside the Bunch&ndash;Davies vacuum

    $$ \sqrt{2k} A_\lambda(t,k) \sim e^{-i \eta k}, \quad  a(t) \sqrt{\frac{2}{k}}\dot{A}_\lambda(k, t) \sim -i e^{-i \eta k}, \quad -k\eta \ll 1 $$ 
     
    at a time implicitly defined by the condition $k = 10^{5/2} k_{\rm UV}(t_{\rm ini})$, with the default $k_{\rm UV}(t) = k_{\rm h}(t)$ obtained from the GEF solution.
    At times $t < t_{\rm ini}$ the mode is assumed to be in Bunch&ndash;Davies. The phase $ e^{-i \eta k}$ is computed separately.

    The mode equations are solved with an explicit Runge&ndash;Kutta of order 5(4), which is implemented in `scipy.integrate.solve_ivp`.

    For creating a custom subclass of `BaseModeSolver` with user-specified  mode equation and initial conditions, you can use the class factory `ModeSolver`.
    
    Attributes
    ----------
    ode_kwargs : dict
        stores the time-dependent background parameters used by `mode_equation` (default keys: 'a', 'H' and 'xi')
    bd_kwargs : dict
        stores the time-dependent background parameters used by `initialise_in_bd` (default: empty)
    cutoff : dict
        the name of $k_{\rm UV}(t)$ in the GEF solution (default: 'kh')
    necessary_keys : set of str
        necessary keys expected as names of `GEFF.bgtypes.Val` instances belonging to the `GEFF.bgtypes.BGSystem` passed when initialising the class
        (default keys: 't', 'N', 'H', 'xi', 'a')
    atol : float
        the default absolute tolerance for `scipy.integrate.solve_ivp`
    """

    ode_kwargs = {"a":None, "H":None, "xi":None}
    bd_kwargs = {}
    cutoff = "kh"
    atol=1e-3

    mode_equation = staticmethod(mode_equation_classic)
    initialise_in_bd = staticmethod(bd_classic)

    necessary_keys = set(["t", "N"] + list(ode_kwargs.keys()) + list(bd_kwargs.keys()) + [cutoff])

    def __init__(self, sys : BGSystem):
        """
        Import the evolution of the background dynamics to configure the solver.

        All values in the BGSystem are treated in numerical units.

        Parameters
        ----------
        sys : BGSystem
            describes the background evolution
        
        Raises
        ------
        KeyError:
            if `sys` is missing a `Val` object from `necessary_keys`
        """
        #Check that all necessary keys are there:
        for key in self.necessary_keys:
            try:
                assert key in sys.value_names()
            except AssertionError:
                KeyError(f"'sys' needs to own an attribute called '{key}'.")
        
        #Ensure that all values from the BGSystem are imported without units
        sys.set_units(False)

        #store the relevant background evolution parameters
        self.__t = sys.t.value
        self.__N = sys.N.value
        kh = getattr(sys, self.cutoff)
        a = np.exp(self.__N)

        self.__khf = CubicSpline(self.__t, kh)

        #import the values for the mode equation and interpolate
        for key in self.ode_kwargs.keys(): 
            val = getattr(sys, key)
            if isinstance(val, Val):
                self.ode_kwargs[key] = CubicSpline(self.__t, val)
        
        #import the values for the mode equation and interpolate
        for key in self.bd_kwargs.keys(): 
            val = getattr(sys, key)
            if isinstance(val, Val):
                self.bd_kwargs[key] = CubicSpline(self.__t, val)
        
        #compute the evolution of conformal time for the phases
        self.__af = CubicSpline(self.__t, a)
        def deta(t, y): return 1/self.__af(t)
        
        soleta = solve_ivp(deta, [min(self.__t), max(self.__t)], np.array([0]), t_eval=self.__t)

        self.__eta = soleta.y[0,:]

        #find lowest t value corresponding to kh(t) = 10^4 kh(0)
        self.__tmin = self.__t[np.where(kh >= 10**4*kh[0])][0]
        
        return
    
    def compute_spectrum(self, nvals : int, t_interval : tuple|NoneType=None, **SolverKwargs) -> GaugeSpec:
        r"""
        Evolve a gauge-field spectrum from Bunch-Davies initial conditions.

        Evolve the mode functions $A_\lambda(t,k)$ and its derivative in time for $n$ modes between $k_{\rm UV}(t_{\rm min})$ and $k_{\rm UV}(t_{\rm max})$.
        The $n$ evolved modes are more densly spaced when $\log k_{\rm UV}(t)$ increases more slowly to ensure a higher density of modes
        crossing the horizon when backreaction effects are relevant.

        Parameters
        ----------
        nvals : int
           The number of modes $n$
        t_interval : tuple or None
            set $t_{\rm min}$ and $t_{\rm max}$. If None, $t_{\rm min}$ is given by $10^4 k_{\rm UV}(t_{min}) = k_{\rm UV}(0)$ and $t_{\rm max} = \max t$.
        **SolverKwargs : kwargs
            tolerance parameters`atol` and `rtol` passed to `solve_ivp` (default: `atol=self.atol`, `rtol=1e-5`)

        Returns
        -------
        spec : GaugeSpec 
            the gauge-field spectrum
        """

        if t_interval is None:
            t_interval = (self.__tmin, max(self.__t))
        ks, tstart = self._find_tinit_bd(self._create_k_array(nvals, t_interval), mode="k")

        modes = np.array([self._evolve_from_bd(k, tstart[i], **SolverKwargs)
                  for i, k in enumerate(ks)])
        
        spec = GaugeSpec({"t":self.__t, "N":self.__N, "k":ks,
                    "Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        return spec
    
    def update_spectrum(self, spec : GaugeSpec, tstart : float, **SolverKwargs) -> GaugeSpec:
        r"""
        Update an existing gauge-field spectrum starting from $t_{\rm start}$

        Starting from the modes stored in `GaugeSpec`, the function re-evaluates the evolution starting from $t_{\rm start}$.
        Additional gauge-field modes are evolved starting from Bunch&ndash;Davies to account for new modes crossing the horizon at times beyond the original range covered by the input spectrum.

        Parameters
        ----------
        spec : GaugeSpec
           the spectrum which is to be updated
        tstart : float
            the starting time $t_{\rm start}$
        **SolverKwargs : kwargs
            as in `compute_spectrum`

        Returns
        -------
        spec : GaugeSpec 
            the updated gauge-field spectrum
        """
        
        indstart = np.where(tstart <= self.__t)[0][0]
        teval = self.__t[indstart:]
        Neval = self.__N[indstart:]
        
        tend = teval[-1]
        indstart = np.where(spec["t"]<teval[0])[0][-1]
        startspec = spec.tslice(indstart)
        tstart = startspec["t"]

        #keep mode-evolution from old spectrum for modes with k < 10*kh(tstart)
        old = np.where(spec["k"] < 10*self.__khf(tstart))[0]
        new = np.where(spec["k"] > 10*self.__khf(tstart))[0]

        #Remove modes which need to be renewed from old spectrum:
        spec.remove_momenta(new)

        #add new modes, adjusting for longer time-span:
        #rethink this! It seems like overkill
        n_newmodes = int(len(new)*max((teval[-1] - teval[0])/(max(spec["t"]) - tstart), 1))

        #Update evolution of modes in spec:
        kold, tvac = self._find_tinit_bd(spec["k"][old], mode="k")

        updatespec={"t":teval, "N":Neval, "k":kold}

        modes = []
        for i, k in enumerate(kold):
            if tvac[i] > teval[0]:
                modes.append( self._evolve_from_bd(k, tvac[i], **SolverKwargs) )
            else:
                yini = np.array(
                            [startspec["Ap"][i].real, startspec["dAp"][i].real,
                            startspec["Ap"][i].imag, startspec["dAp"][i].imag,
                            startspec["Am"][i].real, startspec["dAm"][i].real,
                            startspec["Am"][i].imag, startspec["dAm"][i].imag]
                            )
                
                modes.append( self._evolve_mode(tstart, yini, k, teval, **SolverKwargs) ) 
        
        modes = np.array(modes)

        updatespec.update({"Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        spec.merge_spectra(GaugeSpec(updatespec))

        if n_newmodes > 0:
            newspec = self.compute_spectrum(n_newmodes, t_interval=(tstart, tend))
            #Add new modes
            spec.add_momenta(newspec)
        
        return spec
    
        """
        def EvolveSpectrum(self, spec : GaugeSpec, Nstart, Nstep : float=0.1, atol : float|None=None, rtol : float=1e-5) -> GaugeSpec:
            Neval = np.arange(Nstart, max(self.__N), Nstep)
            teval = CubicSpline(self.__N, self.__t)(Neval)

            indstart = np.where(spec["N"]<Nstart)[0][-1]
            startspec = spec.tslice(indstart)

            klen = len(spec["k"])

            vecode = np.vectorize(lambda t, y, k: self.mode_equation(t, y, k, **self.ode_kwargs),
                                    excluded={0, "t"},
                                signature="(8,n),(n)->(8,n)",
                                )
            def ode(t, y):
                #(k,8) to reshape correctly
                #transposing to match signature of vecode
                y = y.reshape(klen,8).T
                #transposing result s.t. dydt.shape=(k,8)
                dydt  = vecode(t, y, spec["k"]).T
                #reshape is correct again
                return dydt.reshape(8*klen)
            
        spec.merge_spectra(GaugeSpec(newspec))

            return spec
        """
    

    def _evolve_from_bd(self, k : float, tstart : float,
                    atol : float|None=None, rtol : float=1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evolve a mode with momentum $k$ starting from Bunch-Davies initial conditions.

        Parameters
        ----------
        k : float
           the momentum $k$
        tstart : float
            the corresponding initialisation time, $t_{\rm init}$
        atol : float or None
            `atol` used by `solve_ivp` if None, use `self.atol`
        rtol : float
            `rtol` used by `solve_ivp`

        Returns
        -------
        yp : NDArray
            the positive helicity mode
        dyp : NDArray
            the derivative of the positive helicity mode
        ym : NDArray
            the positive helicity mode
        dym : NDArray
            the derivative of the negative helicity mode
        """

        #Initial conditions for y and dydt for both helicities (rescaled appropriately)
        yini = self.initialise_in_bd(tstart, k, **self.bd_kwargs)

        teval = self.__t

        istart = np.where(teval>tstart)[0][0]

        yp, dyp, ym, dym = self._evolve_mode(tstart, yini, k, teval[istart:], atol, rtol)

        #conformal time needed for relative phases
        eta = self.__eta
        
        #the mode was in vacuum before tstart
        yvac = np.array([self.initialise_in_bd(t, k, **self.bd_kwargs) for t in teval[:istart]]).T 
        phasevac = (np.exp(-1j*k*eta[:istart]))
        vac = yvac * phasevac

        #Create array of mode evolution stringing together vacuum and non-vacuum time evolutions to get evolution from t0 to tend
        yp = np.concatenate([(vac[0,:] + 1j*vac[2,:]), yp*np.exp(-1j*k*eta[istart])])
        dyp = np.concatenate([(vac[1,:] + 1j*vac[3,:]), dyp*np.exp(-1j*k*eta[istart])])
        ym = np.concatenate([(vac[4,:] + 1j*vac[6,:]), ym*np.exp(-1j*k*eta[istart])])
        dym = np.concatenate([(vac[5,:] + 1j*vac[7,:]), dym*np.exp(-1j*k*eta[istart])])

        return yp, dyp, ym, dym
    
    def _evolve_mode(self, tini : float, yini : np.ndarray, k : float, teval : np.ndarray,
                    atol : float|None=None, rtol : float=1e-5):
        """
        Evolve a mode of momentum $k$ from $t_{\rm ini}$.

        Parameters
        ----------
        tini : float
           the initial time $t_{\rm ini}$
        yini : NDArray
            (8,) array containing the initial data
        k : float
            the momentum $k$
        teval : NDArray
            the times at which the returned solution is evaluated
        atol : float or None
            `atol` used by `solve_ivp` if None, use `self.atol`
        rtol : float
            `rtol` used by `solve_ivp`

        Returns
        -------
        yp : NDArray
            the positive helicity mode
        dyp : NDArray
            the derivative of the positive helicity mode
        ym : NDArray
            the positive helicity mode
        dym : NDArray
            the derivative of the negative helicity mode
        """
        #Define ODE
        def ode(t, y): return self.mode_equation(t, y, k, **self.ode_kwargs)

        if atol is None:
            atol = self.atol
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tini, max(teval)], yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)

        yp = (sol.y[0,:] + 1j*sol.y[2,:])
        dyp = (sol.y[1,:] + 1j*sol.y[3,:])
        ym = (sol.y[4,:] + 1j*sol.y[6,:])
        dym = (sol.y[5,:] + 1j*sol.y[7,:])

        return yp, dyp, ym, dym 
        

    def _create_k_array(self, nvals : int, t_interval : tuple) -> np.ndarray:
        """
        Create an array of $n$ momenta

        The $n$ modes are generated between $k_{\rm UV}(t_{\rm min})$ and $k_{\rm UV}(t_{\rm max})$.
        First, $m$ of modes $k = k_{\rm UV}$ with t evenly spaced between $t_{\rm min}$ and $t_{\rm max}$ are generated.
        As $k_{\rm UV}$ is monotonic but not strictly monotonic, $m\leq n$. To fill up to $n$ modes, $n-m$ additional modes are
        created between the existing times $(t_{i},t_{i+1})$ moving backwards from $t_{\rm max}$ to favour larger momenta. 

        Parameters
        ----------
        nvals : int
            The size of the output array
        t_interval : tuple
            sets $t_{\rm min}$ and $t_{\rm max}$

        Returns
        -------
        NDArray
            an array of momenta with size nvals
        """

        #create an array of values log(10*kh(t))
        logks = np.round( np.log(10*self.__khf(np.linspace(t_interval[0], t_interval[1], nvals))), 3)
        #filter out all values that are repeating (kh is not strictly monotonous)
        logks = np.unique((logks))

        #fill up the array of ks values with additional elements between gaps, favouring larger k
        while len(logks) < nvals:
            num_newk = nvals - len(logks)
            if num_newk >= len(logks):
                newvals = (logks[1:] + logks[:-1])/2
            else:
                newvals = (logks[-num_newk:] + logks[-num_newk-1:-1])/2
            logks = np.sort(np.concatenate([logks, newvals]))
        return np.exp(logks)
    
    def _find_tinit_bd(self, init : np.ndarray, mode : str="k") -> Tuple[np.ndarray, np.ndarray]:
        """
        Determines the pair of $k$ and $t$ satisfying $k = 10^(5/2)*k_h(t)$.

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

        if mode=="t":
            tstart = init
            ks = 10**(5/2)*self.__khf(tstart)

        elif mode=="k":
            ks = init
            
            tstart = []
            for k in ks:
                ttmp  = self.__t[np.where(k >= 10**(5/2)*self.__khf(self.__t))[0][-1]]
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(self.__N, self.__t)(init)
            ks = 10**(5/2)*self.__khf(tstart)

        else:
            raise KeyError("'mode' must be 't', 'k' or 'N'")

        return ks, tstart
        
   
    
    
def ModeSolver(new_mode_eq : Callable, ode_keys : list[str], new_bd_init : Callable, init_keys : list[str], new_cutoff : str="kh", default_atol : float=1e-3):
    r"""
    Create a subclass of `BaseModeSolver` with custom mode equation and initial conditions.

    In case your GEF model does not follow the pre-defined gauge-field mode equation defined for `BaseModeSolver`, 
    you can create a subclass of it by defining new methods for `mode_equation` and `initialise_in_bd` through `new_mode_eq` and `new_bd_init`.
    
    The method `new_mode_eq` needs obey the following conditions:
    1. The call signature is `f(t,y,k,**kwargs)`
    2. The arguments `t` / `k` expect floats representing time / momentum
    3. The argument `y` expects a `numpy.ndarrray` of shape (8,) with indices
        -  0 & 2 / 4 & 6: real & imag. part of $\sqrt{2k} A_\lambda(t_{\rm init},k)$ for $\lambda = 1$ / $\lambda = -1$
        -  1 & 3 / 5 & 7: real & imag. part of $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$ for $\lambda = 1$ / $\lambda = -1$
    4. The kwargs are functions of the argument `t`.
    5. The return is the time derivative of `y`

    The method `new_bd_Init` needs to obey the following conditions:
    1. The call signature is `f(t,k,**kwargs)`
    2. The arguments `t` / `k` expect floats representing time / momentum
    3. The kwargs are functions of the argument `t`.
    4. The return is a `numpy.ndarrray` of shape (8,)  with indices
        -  0 & 2 / 4 & 6: real & imag. part of $\sqrt{2k} A_\lambda(t_{\rm init},k)$ for $\lambda = 1$ / $\lambda = -1$
        -  1 & 3 / 5 & 7: real & imag. part of $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$ for $\lambda = 1$ / $\lambda = -1$
    
    The lists `ode_keys` and `init_keys` are handled as follows:
    - `ode_keys` and `init_keys` need to contain the keys associated to the respective kwargs of `new_mode_eq` and `new_bd_init`.
    - These keys correspond to names of `GEFF.bgtypes.Val` objects belonging to a `GEFF.bgtypes.BGSystem` passed to the class upon initialisation.
        The respective `Val` objects are interpolated to obtain functions of time. 
        These functions are then passed to to the corresponding keyword arguments of `new_mode_eq`  and `new_bd_init`.
    - `ode_keys` and `init_keys` are added to the `necessary_keys` attribute of the new subclass.

    You can also overwrite the `cuttoff` and `atol` inherited from `BaseModeSolver` 

    Parameters
    ----------
    new_mode_eq : Callable
        a new mode equation
    ode_keys : list of str
        the non-standard keywords of `new_mode_eq`
    new_bd_init : Callable
        a new mode bd initial condition
    init_keys : list of str
        the non-standard keywords of `new_bd_init`
    new_cutoff : str
        the new `cutoff` attribute of the subclass
    default_atol : float
        the default absolute tolerance used by the subclass

    Returns
    -------
    NewModeSolver : class
        the newly defined subclass of `BaseModeSolver`

    Example
    -------
    ```python
        import numpy as np
        from GEFF.bgtypes import BGSystem, BGVal

        # Define a new mode equation:
        def custom_mode_eq(t, y, k, a, X, Y):
            #create a return array of the right shape
            dydt = np.ones_like(y)

            #compute real-part time derivatives for positive modes
            dydt[0] = k / a(t) * y[1] # a is a function of t.
            dydt[1] = X(t)/Y(t)*y[0] # X and Y are functions of t.

            #compute imag-part time derivatives for positive modes
            ...

            #compute real-part time derivatives for negative modes
            ...
            ...

            return dydt

        # Define a new initial condition for the modes:
        def custom_bd_init(t, k, alpha):
            y = alpha(t)*np.array([...]) # alpha is a function of t.
            return y

        # the kwargs of custom_mode_eq are 'a', 'X' and 'Y':
        custom_ode_keys = ['a', 'X', 'Y']

        # the kwarg of custom_bd_init is 'alpha':
        custom_init_keys = ['alpha']

        # Define the custom mode solver using the class factory:
        CustomModeSolver = ModeSolver(custom_mode_eq, custom_ode_keys,
                                         custom_bd_init, custom_init_keys)

        # To initialise CustomModeSolver we need a BGSystem. 
        # Its Val instances need to have the right names however:
        # The default: 't', 'N', 'kh' (we did not rename 'cutoff')
        t = BGVal("t", -1, 0)
        N = BGVal("N", 0, 0)
        kh = BGVal("kh", 1, 0)
        # Because of custom_mode_eq we need 'a', 'X', 'Y'
        a = BGVal("a", 0, 0)
        X = BGVal("X", 2, 0)
        Y = BGVal("X", 2, 0)
        # Because of custom_bd_init we need 'alpha'
        alpha = BGVal("alpha", 0, 0)

        # When in doubt, consult necessary_keys:
        print(CustomModeSolver.necessary_keys)

        # We create the BGSystem and initialise all its values:
        sys = BGSystem({t, N, kh, a, X, Y, alpha}, 1e-6, 1)
        sys.initialise(t)(...)
        ...

        # The values in sys can now be used to initialise CustomModeSolver
        MbM = CustomModeSolver(sys)

        # Let's compute a spectrum using the new setup:
        MbM.compute_spectrum(100)
        ```
    """
    
    class ModeSolver(BaseModeSolver):
        """
        A subclass of BaseModeSolver new mode equation and initial condition adapted to a new GEF model

        It Inherits all methods from 'BaseModeSolver' but changes the following class attributes
            - mode_equation
            - ode_kwargs
            - initialise_in_bd
            - init_kwargs
            - cutoff
            - atol
            
        This entails that 'compute_spectrum' will now evolve modes according to initialise_in_bd and ModeEom.
        """
        
        #Overwrite class attibutes of ModeByMode with new mode equations, boundary conditions and default tolerances.
        mode_equation = staticmethod(new_mode_eq)
        ode_kwargs = dict(zip(ode_keys, [None for x in ode_keys]))

        initialise_in_bd = staticmethod(new_bd_init)
        bd_kwargs = dict(zip(init_keys, [None for x in init_keys]))

        atol=default_atol
    
    return ModeSolver










    
    
    
    
    
    
    
    
    