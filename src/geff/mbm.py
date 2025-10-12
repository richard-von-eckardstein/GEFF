import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.integrate import quad, simpson
from .bgtypes import Variable, Constant, Func, BGSystem
from .utility.mode  import mode_equation_classic, bd_classic
from typing import Tuple, Callable, ClassVar
from types import NoneType
from ._docs import generate_docs, docs_mbm
from tabulate import tabulate


class GaugeSpec(dict):
    def __init__(self, in_dict : dict):
        """
        Initialise the spectrum from a dictionary.

        Parameters
        ----------
        in_dict : dict
            dictionary with keys `'t'`, `'N'`, `'k'`, `'Ap'`, `'dAp'`, `'Am'`, `'dAm'`

        Raises
        ------
        KeyError
            if one of the necessary keys is missing.
        ValueError
            if `len(in_dict['t']) != len(in_dict['N'])` or if
            `in_dict['X']).shape != (len(in_dict['k']),len(in_dict['t']))` for `'X'` in `['Ap', 'dAp', 'Am', 'dAm']`.
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
        Compute the three integrals $\mathcal{F}_\mathcal{X}^{(n)}(t)$ for $\mathcal{X} = \mathcal{E}, \mathcal{B},\mathcal{G}$ for fixed $n$ and each time $t$ in the spectrum.

        If the time coordinates stored in `BG` do not match those stored in the spectrum, $k_{\rm UV}(t)$ is evaluated using interpolation.

        Parameters
        ----------
        BG : BGSystem
            a system containing the UV cut-off, $k_{\rm UV}(t)$
        n : int
            the integer $n$
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

        ind = np.searchsorted(self["t"], spec["t"][0], "left")

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
        units = BG.units
        BG.units = (False)

        scale = getattr(BG, cutoff)

        bl, mask = self._check_overlap(BG.t)

        if bl:
            self["cut"] = scale[mask]
        else:
            self["cut"] = CubicSpline(BG.t, scale)(self["t"])
        
        BG.units = (units)

        return self["cut"]
    
    def _get_reference(self, BG : BGSystem, references=["E", "B", "G"], cutoff="kh"): 
        units = BG.units
        BG.units = (False)

        scale = getattr(BG, cutoff)

        bl, mask = self._check_overlap(BG.t)

        Fref = []
        for val in references: 
            val_arr = (getattr(BG, val)*(np.exp(BG.N)/scale)**4)
            if bl:
                Fref.append( val_arr[mask] )
            else:
                Fref.append( CubicSpline(BG.t, val_arr)(self["t"]) ) 

        BG.units = (units)

        return Fref
    
    
    def _estimate_error(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                         **IntegratorKwargs):
        FMbM = self.integrate(BG, n=0, cutoff=cutoff, **IntegratorKwargs)
        Fref = self._get_reference(BG, references, cutoff)

        errs = []

        for i, spl in enumerate(Fref):
            err =  np.minimum(1e3, abs( 1 - (FMbM[:,i,0]+1e-20)/(spl+1e-20) )) #avoid divide by zeros
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
        print("The mode-by-mode comparison finds the following relative deviations from the GEF solution:\n")

        lst = []
        for err in bin_errs:
            rmserr = 100*np.sqrt(np.sum(err**2)/len(err))
            finerr = 100*err[-1]
            maxerr = 100*max(err)

            tmaxerr = bin_terr[np.argmax(err)]
            tfinerr = bin_terr[-1]

            lst.append([f"{maxerr:.1f}% at {tmaxerr:.1f}",
                        f"{finerr:.1f}% at {tfinerr:.1f}",
                        f"{rmserr:.1f}%"])

        print(tabulate(lst, headers=["max", "final", "RMS"], showindex=references, tablefmt="simple")+"\n")
        return

   
# continue from here next time
class SpecSlice(dict):    

    def __init__(self, in_dict):
        super().__init__(in_dict)

    def integrate_slice(self, n : int=0, integrator="simpson", modethr : int=100,
                         epsabs : float=1e-20, epsrel : float=1e-4, interpolator=PchipInterpolator
                         )-> Tuple[np.ndarray, np.ndarray]:
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
            if integrator=="simpson":
                msk = np.where(x < 0)[0]
                if len(msk) < modethr: #cannot trust simpsons integration for too few modes.
                    return res
                x = x[msk]
                res[i,0] = (self._simpson_integrate( integs[i,0,msk] ,x) 
                                         + (-1)**n*self._simpson_integrate(integs[i,1,msk], x) )
                res[i,1] = 1e-6*res[i,0]

            elif integrator=="quad":
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
    _ode_dict : dict = {"a": "a", "H":"H", "xi":"xi"}
    _bd_dict = {}

    cutoff : ClassVar[str] = "kh"
    r"""The name of $k_{\rm UV}(t)$ in the GEF solution."""
    
    atol : ClassVar[float] = 1e-3
    """The default absolute tolerance used in `scipy.integrate.solve_ivp`"""

    necessary_keys : ClassVar[set] = (["t", "N", "a", "H", "xi"] + [cutoff])
    """The class expects these attributes in the `.bgtypes.BGSystem` passed on initialisation."""

    mode_equation = staticmethod(mode_equation_classic)
    initialise_in_bd = staticmethod(bd_classic)

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
            if `sys` is missing a `Val` or `Func` object from `necessary_keys`
        ValueError:
            if the keys in `necessary_keys` are not `Val` or `Func` objects.
        """
        og_units = sys.units
        sys.units = False

        #Check that all necessary keys are there:
        for key in self.necessary_keys:
            check = True
            check = ( (key in sys.variable_names())
                        or
                        (key in sys.constant_names())
                        or 
                        (key in sys.function_names())
                    )
            if not(check):
                raise KeyError(f"'sys' needs to own an attribute called '{key}'.")

        #store the relevant background evolution parameters
        self._t = sys.t.value
        self._N = sys.N.value
        kh = getattr(sys, self.cutoff)
        a = np.exp(self._N)

        self._khf = CubicSpline(self._t, kh)

        self._kwargs_ode = {kwarg:None for kwarg in self._ode_dict.values()}
        self._kwargs_bd = {kwarg:None for kwarg in self._bd_dict.values()}

        #import the values for the mode equation and interpolate
        for key in self.necessary_keys: 
            obj = getattr(sys, key)
            if isinstance(obj, Variable):
                value = getattr(sys, key).value
                func = CubicSpline(self._t, value)
            elif isinstance(obj, Func):
                arg_vals = []
                for arg in obj.args:
                    arg_vals.append(getattr(sys, arg.name))
                value = obj(*arg_vals)
                func =  CubicSpline(self._t, value)
            elif isinstance(obj, Constant):
                value = getattr(sys, key).value
                func =  CubicSpline(self._t, value*np.ones_like(self._t))
            else:
                raise ValueError(f"'{key}' should refer to either a 'Val' or 'Func' subclass.")
                
            if key in self._ode_dict.keys():
                kwarg = self._ode_dict[key]
                self._kwargs_ode[kwarg] = func
            if key in self._bd_dict.values():
                kwarg = self._bd_dict[key]
                self._kwargs_bd[kwarg] = func
        
        #compute the evolution of conformal time for the phases
        af = CubicSpline(self._t, a)
        def deta(t, y): return 1/af(t)
        
        soleta = solve_ivp(deta, [min(self._t), max(self._t)], np.array([0]), t_eval=self._t)

        self.__eta = soleta.y[0,:]

        #find lowest t value corresponding to kh(t) = 10^4 kh(0)
        self.__tmin = self._t[np.searchsorted(kh, 10**4*kh[0], "right")]

        sys.units = og_units
        
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
            t_interval = (self.__tmin, max(self._t))
        ks, tstart = self._find_tinit_bd(self._create_k_array(nvals, t_interval), mode="k")

        modes = np.array([self._evolve_from_bd(k, tstart[i], **SolverKwargs)
                  for i, k in enumerate(ks)])
        
        spec = GaugeSpec({"t":self._t, "N":self._N, "k":ks,
                    "Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        return spec
    
    def update_spectrum(self, spec : GaugeSpec, tstart : float, **SolverKwargs) -> GaugeSpec:
        r"""
        Update an existing gauge-field spectrum starting from $t_{\rm start}$

        Starting from the modes stored in `GaugeSpec`, the function re-evaluates the evolution starting from $t_{\rm start}$.
        Additional gauge-field modes are evolved starting from Bunch&ndash;Davies to account for new modes crossing the horizon
          at times beyond the original range covered by the input spectrum.

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
        
        indstart = np.searchsorted(self._t, tstart, "left")
        teval = self._t[indstart:]
        Neval = self._N[indstart:]
        
        tend = teval[-1]
        indstart = np.searchsorted(spec["t"], teval[0], "left")
        startspec = spec.tslice(indstart)
        tstart = startspec["t"]

        #keep mode-evolution from old spectrum for modes with k < 10*kh(tstart)
        old = np.where(spec["k"] < 10*self._khf(tstart))
        new = np.where(spec["k"] >= 10*self._khf(tstart))

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

            vecode = np.vectorize(lambda t, y, k: self.mode_equation(t, y, k, **self._ode_kwargs),
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
        yini = self.initialise_in_bd(tstart, k, **self._kwargs_bd)

        teval = self._t

        istart = np.searchsorted(teval, tstart, "right")

        yp, dyp, ym, dym = self._evolve_mode(tstart, yini, k, teval[istart:], atol, rtol)

        #conformal time needed for relative phases
        eta = self.__eta
        
        #the mode was in vacuum before tstart
        yvac = np.array([self.initialise_in_bd(t, k, **self._kwargs_bd) for t in teval[:istart]]).T 
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
        def ode(t, y): return self.mode_equation(t, y, k, **self._kwargs_ode)

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
        r"""
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
        logks = np.round( np.log(10*self._khf(np.linspace(t_interval[0], t_interval[1], nvals))), 3)
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
        Determines the pair of $k$ and $t$ satisfying $k = 10^(5/2)k_h(t)$.

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
            ks = 10**(5/2)*self._khf(tstart)

        elif mode=="k":
            ks = init
            
            tstart = []
            for k in ks:
                ttmp  = self._t[np.searchsorted(10**(5/2)*self._khf(self._t), k, "right")]
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(self._N, self._t)(init)
            ks = 10**(5/2)*self._khf(tstart)

        else:
            raise KeyError("'mode' must be 't', 'k' or 'N'")

        return ks, tstart
        
    
def ModeSolver(new_mode_eq : Callable, ode_keys : list[str], new_bd_init : Callable, init_keys : list[str], new_cutoff : str="kh", default_atol : float=1e-3):
    class CustomModeSolver(BaseModeSolver):
        """
        A subclass of `BaseModeSolver` new mode equation and initial condition adapted to a new GEF model

        It Inherits all methods from `BaseModeSolver` but changes the attributes:
        - `BaseModeSolver.mode_equation`
        - `BaseModeSolver.initialise_in_bd`
        - `BaseModeSolver.necessary_keys`
        - `BaseModeSolver.cutoff`
        - `BaseModeSolver.atol`
            
        This entails that `compute_spectrum` will now evolve modes according to `initialise_in_bd` and `mode_equation`.
        """
        
        #Overwrite class attibutes of ModeByMode with new mode equations, boundary conditions and default tolerances.
        mode_equation = staticmethod(new_mode_eq)
        _ode_dict = {attr.name:kwarg for kwarg, attr in ode_keys.items()}

        initialise_in_bd = staticmethod(new_bd_init)
        _bd_dict = {attr.name:kwarg for kwarg, attr in init_keys.items()}
        cutoff = new_cutoff

        necessary_keys = list( {"t", "N", new_cutoff}.union(set(_ode_dict.keys())).union(set(_bd_dict.keys())) )

        atol=default_atol

    CustomModeSolver.__qualname__ = "ModeSolver"
    CustomModeSolver.__module__ = __name__
    return CustomModeSolver


#define longer method docs from docs_mbm:
generate_docs(docs_mbm.DOCS)







    
    
    
    
    
    
    
    
    