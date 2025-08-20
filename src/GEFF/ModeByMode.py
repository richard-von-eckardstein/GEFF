import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.integrate import quad, simpson

from GEFF.bgtypes import Val, BGSystem
from GEFF.Models.EoMsANDFunctions.ModeEoMs import ModeEoMClassic, BDClassic

from numpy.typing import NDArray
from typing import Tuple, Callable

class GaugeSpec(dict):
    """
    A class representing a spectrum of gauge-field modes as a function of time.
    This class inherits from 'dict', and necessarily requires the following keys:
    't', 'N', 'k', 'Ap', 'dAp', 'Am', 'dAm'

    Attributes
    ----------
    t : NDArray
        an array of physical times at which the spectrum is known
    N : NDArray
        an array of e-folds associated to t
    k : NDarray
        an array of wavenumbers k for which the spectrum is known
    Ap, Am : NDarray
        arrays of shape (len(k), len(t)) containing the mode functions sqrt(2*k)*A_\\pm(k, t)
    dAp, dAm : NDarray
        arrays of shape (len(k), len(t)) containing the mode-function derivatives  sqrt(2/k)*e^N*dA_\\pm(k, t)/dt

    Class Methods
    -------------
    read_spec
        Initialise the class from data stored in a file.

    Methods
    -------
    save_spec()
        Store the spectrum in a file.
    get_dim()
        Retrieve the number of modes and times encoded in the spectrum 
    tslice()
        Retrieve the spectrum at a moment of time
    kslice()
        Retrieve the spectrums evolution for a fixed wavenumber

    Example
    -------
    >>> spec = GaugeSpec.read_spec(somefile)
    >>> slice = spec.tslice(100) #return the spectrum at spec["t"][100]
    >>> print(f"This is the spectrum of positive-helicity modes at time {slice['t']}:)
    >>> print(slice["Ap"]})
    """

    def __init__(self, modedic):
        #initialise spectrum as a dictionary
        for key in modedic.keys():
            assert key in ["t", "N", "k", "Ap", "dAp", "Am", "dAm"]

        assert len(modedic["t"]) == len(modedic["N"])
        for key in ["Ap", "dAp", "Am", "dAm"]:
            assert modedic[key].shape == (len(modedic["k"]), len(modedic["t"]))
        super().__init__(modedic)

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
        GaugeSpec
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
    
    def save_spec(self, path : str, thinning = 5):
        """
        Store the spectrum in a file.

        Parameters
        ----------
        path : str
            path to the data 
        """
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
        Retrieve the spectrums evolution for a fixed wavenumber

        Returns
        -------
        dict
            a dictionary encoding the number of modes 
            and the number of times stored in the spectrum
        """
        return {"kdim":len(self["k"]), "tdim":len(self["t"])}
    
    def tslice(self, ind : int) -> dict:
        """
        Retrieve the spectrum at a moment of time

        Parameters
        ----------
        ind : int
            the index corresponding to the time at which to retrieve the spectrum

        Returns
        -------
        dict
            a dictionary with keys like self.
        """

        spec_slice = {}
        for key in self.keys():
            if key in ["N", "t", "cut"]:
                spec_slice[key] = self[key][ind]
            elif key=="k":
                spec_slice[key] = self[key]
            else:
                spec_slice[key] = self[key][:,ind]
        return GaugeSpecSlice(spec_slice)
    
    def kslice(self, ind : int) -> dict:
        """
        Retrieve the spectrum for a fixed wavenumber

        Parameters
        ----------
        ind : Int
            the index corresponding to the wavenumber at which to retrieve the spectrum

        Returns
        -------
        dict
            a dictionary with keys like self.     
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
    
    def merge_spectra(self, spec):
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
    
    def add_momenta(self, spec):
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
    
    def integrate(self, BG : BGSystem, n : int=0, cutoff="kh", **IntegratorKwargs) -> NDArray:
        """
        Integrate an input spectrum to determine the expectation values of (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)

        Parameters
        ----------
        spec : GaugeSpec
            the spectrum to be integrated
        n : int
            the power in curls in the expectation value, i.e. (E rot^n E).
        epsabs : float
            absolute tolerance used by scipy.integrate.quad
        epsrel : float
            relative tolerance used by scipy.integrate.quad 

        Returns
        -------
        NDArray
            an array of shape (len(spec["t"]), 3) corresponding to (E, rot^n E), (B, rot^n B), (E, rot^n B)
        """

        self._add_cutoff(BG, cutoff)

        tdim = self.get_dim()["tdim"]

        FMbM = np.zeros((tdim, 3,2))
        for i in range(tdim):
            spec_slice = self.tslice(i)
            FMbM[i,:] = spec_slice.integrate_slice(n=n, **IntegratorKwargs)
        
        return FMbM
    
    def _estimate_error(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                         **IntegratorKwargs):
        FMbM = self.integrate(BG, n=0, cutoff=cutoff, **IntegratorKwargs)
        Fref = self._get_reference(BG, references, cutoff)

        errs = []

        for i, spl in enumerate(Fref):
            err =  np.minimum(1e3, abs( 1 - FMbM[:,i,0]/ spl ))
            errs.append(np.where(np.isnan(err), 10.0, err))

        return errs
    
    def _process_error(self, errs, terr, errthr):
        removals = []
        for err in errs:
            #remove the first few errors where the density of modes is low:
            removals.append(np.where(err < errthr)[0][0])
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

    def estimate_GEF_error(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                                    errthr=0.025, binning=5, verbose : bool=True,
                                    **IntegratorKwargs) -> Tuple[list, NDArray]:
        """
        Estimate the relative deviation in E^2, B^2, E.B between a GEF solution and a mode-spetrum as a function of e-folds.

        Parameters
        ----------
        spec : GaugeSpec
            the spectrum against which to compare the GEF results.
        epsabs : float
            absolute tolerance used by scipy.integrate.quad
        epsrel : float
            relative tolerance used by scipy.integrate.quad 

        Returns
        -------
        errs : list
            a list of estimated errors, each index corresponding to E^2, B^2, E.B respectively
        Nerr : NDArray
            an array of e-fold-bins to which the errors in errs are associated.
        """

        og_errs = self._estimate_error(BG, references, cutoff, **IntegratorKwargs)
        terr = self["t"]

        if binning is not None:
            errs, terr = self._bin_error(og_errs, binning)
        else:
            errs = og_errs
        
        errs, terr = self._process_error(errs, terr, errthr)

        if verbose:
            self._error_summary(errs, terr, references)

        return errs, terr, og_errs
    
class GaugeSpecSlice(dict):
    def __init__(self, modedic):
        super().__init__(modedic)

    def _Espec(self, lam):
        return abs(self["dA"+lam])**2
    
    def _Bspec(self, lam):
        return abs(self["A"+lam])**2
    
    def _Gspec(self, lam):
        return (self["A"+lam].conjugate()*self["dA"+lam]).real
    
    def _simpson_integrate(self, integrand, x):
        integrand = integrand*np.exp(x)
        return simpson(integrand, x)

    def _quad_integrate(self, integrand, x, epsabs : float=1e-20, epsrel : float=1e-4, interp=PchipInterpolator):
        msk = np.where(abs(integrand) > 1e-1*max(epsrel*max(abs(integrand)), epsabs))[0]
        if len(msk) > 0:
            spl = interp(x, np.arcsinh(integrand))
            f = lambda x: np.sinh(spl(x))*np.exp(x)
            val, err = quad(f, x[msk][0], 0., epsabs=epsabs, epsrel=epsrel)
            return np.array([val, err])
        else:
            return np.nan*np.ones((2))
        
        
    def integrate_slice(self, n : int=0, epsabs : float=1e-20, epsrel : float=1e-4, interp=PchipInterpolator,
                            method="simpson", modethr=100) -> Tuple[NDArray, NDArray]:
        """
        Integrate an input spectrum at a fixed time t to obtain (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)

        Parameters
        ----------
        specAtT : dict
            the spectrum at time t, obtained by GaugeSpec.tslice
            n : int
            the power in curls in the expectation value, i.e. E rot^n E etc.
        epsabs : float
            absolute tolerance used by scipy.integrate.quad
        epsrel : float
            relative tolerance used by scipy.integrate.quad 

        Returns
        -------
        vals : NDArray
            an array of size 3 corresponding to (E, rot^n E), (B, rot^n B), (E, rot^n B)
        errs : NDArray
            the error on vals as estimated by scipy.integrate.quad
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
                resp = self._quad_integrate( integs[i,0,:], x, epsabs, epsrel, interp)
                resm = self._quad_integrate( (-1)**n*integs[i,1,:], x, epsabs, epsrel, interp)
                res[i,:] = resp +resm
            
        res = 1/(2*np.pi)**2*res/(n+4)

        res[:,1] = abs(res[:,1]/res[:,0])
        return res
    
    
def ModeSolver(ModeEq : Callable, EoMkeys : list, BDEq : Callable, Initkeys : list, default_atol : float=1e-3):
    """
    Class-factory creating a custom ModeSolver-class with new mode equations and initial conditions adapted to a modified version of the GEF.

    Parameters
    ----------
    ModeEq : function
        a new mode equation called as ModeEq(t, y, **kwargs)
    EoMKeys : list of str
        a list of parameter names passed to the mode equation.
        These names must match the kwargs of ModeEq.
    BDEQ : function
        a function that initialises modes in Bunch-Davies. The signature must be BDInitEQ(t, k, **kwargs)
    Initkeys : list of str
        a list of parameter names passed to the Bunch-Davies initialiser
        These names must match the kwargs of BDEq.
    default_atol : float
        the default absolute tolerance used by the ModeByMode class

    Returns
    -------
    class
        a modified ModeByMode class adapted to a modified GEF-model.

    Examples
    --------
    >>> def new_mode_eq(t, y, k, a, b, c):
    ...     # Define your new mode equation here
    ...     dydt = np.ones_like(y)
    ...     dydt[0] = ...
    ...     ...
    ...     return dydt
    ...
    >>> def new_bd_init(t, k, alpha, beta):
    ...     # Define your new Bunch-Davies initial conditions here
    ...     return [1, 0, 0, -1, 1, 0, 0, -1]
    ...
    >>> EoMkeys = ["a", "b", "c"]
    >>> Initkeys = ["alpha", "beta"]
    >>> CustomModeSolver = ModeSolver(new_mode_eq, EoMkeys, new_bd_init, Initkeys, default_atol=1e-4)
    >>> solver = CustomModeSolver(values)
    """
    
    class ModeSolver(BaseModeSolver):
        """
        A custom ModeSolver class with new mode equations and initial conditions adapted to a modified version of the GEF
        It Inherits all methods from 'BaseModeSolver' but changes the following class attributes
            - ModeEoM
            - EoMKwargs
            - BDInit
            - InitKwargs
            - default-atol
        This entails that 'compute_spectrum' will now evolve modes according to BDInit and ModeEom.

        Methods
        -------
        compute_spectrum()
            Compute a gauge-field spectrum by evolving each mode in time starting from Bunch-Davies initial conditions
        integrate()
            Integrate an input spectrum to determine the expectation values of (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)
        estimate_GEF_error()
            Estimate the relative deviation in E^2, B^2, E.B between a GEF solution and a mode-spetrum as a function of e-folds

        Example
        -------
        >>> M = ModeSolver(G) #initialise the class by a BGSystem or GEF instance
        ... 
        >>> spec = M.compute_spectrum(500) #compute a gauge-field spectrum of 500 modes from G
        """
        
        #Overwrite class attibutes of ModeByMode with new mode equations, boundary conditions and default tolerances.
        ModeEoM = staticmethod(ModeEq)
        EoMKwargs = dict(zip(EoMkeys, [None for x in EoMkeys]))

        BDInit = staticmethod(BDEq)
        InitKwargs = dict(zip(Initkeys, [None for x in Initkeys]))

        atol=default_atol
    
    return ModeSolver


class BaseModeSolver:
    """
    A class used to solve the gauge-field mode equations for standard axion inflation based on a solution to the GEF equations.

    Methods
    -------
    compute_spectrum()
        Compute a gauge-field spectrum by evolving each mode in time starting from Bunch-Davies initial conditions

    Example
    -------
    >>> M = ModeByMode(G) #initialise the class by a BGSystem or GEF instance
    ... 
    >>> spec = M.compute_spectrum(500) #compute a gauge-field spectrum of 500 modes from G
    >>> errs, Nerr = M.estimate_GEF_error(spec) #asses the agreement between G and spec
    """

    #Rename!
    ModeEoM = staticmethod(ModeEoMClassic)
    EoMKwargs = {"a":None, "H":None, "xi":None}
    BDInit = staticmethod(BDClassic)
    InitKwargs = {}
    atol=1e-3

    def __init__(self, values : BGSystem):
        #Ensure that all values from the GEF are imported without units
        values.set_units(False)

        #store the time values of the GEF
        self.__t = values.t.value
        self.__N = values.N.value
        kh = values.kh
        a = values.a

        self.__khf = CubicSpline(self.__t, kh)

        #import the keys 
        for key in self.EoMKwargs.keys(): 
            val = getattr(values, key)
            if isinstance(val, Val):
                self.EoMKwargs[key] = CubicSpline(self.__t, val)
        
        for key in self.InitKwargs.keys(): 
            val = getattr(values, key)
            if isinstance(val, Val):
                self.InitKwargs[key] = CubicSpline(self.__t, val)

        for key in ["E", "B", "G"]:
            func = CubicSpline(self.__N, (a/kh)**4 * getattr(values, key))
            setattr(self, f"__{key}f", func)
        
        self.__af = CubicSpline(self.__t, a)
        def deta(t, y): return 1/self.__af(t)
        
        soleta = solve_ivp(deta, [min(self.__t), max(self.__t)], np.array([0]), t_eval=self.__t)

        self.__eta = soleta.y[0,:]

        #Nend = G.EndOfInflation()

        maxN = max(self.__N)#min(max(self.__N), Nend)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        self.maxk = CubicSpline(self.__N, 10*kh)(maxN)
        self.mink = 10**4*kh[0]

        #find lowest t value corresponding to kh(t) = 10*mink
        self.__tmin = self.__t[np.where(kh >= self.mink)][0]
        
        return
    
    def compute_spectrum(self, nvals : int, t_interval=None, **SolverKwargs) -> GaugeSpec:
        """
        Compute a gauge-field spectrum by evolving each mode in time starting from Bunch-Davies initial conditions

        Parameters
        ----------
        nvals : int
           The number of modes between self.mink and self.maxk at which to compute the spectrum
        Nstep : float
            the spectrum is stored at times evenly spaced in e-folds with spacing Nstep
        teval : list of float
            physical time points at which the mode function A(t,k,+/-) and its derivatives will be returned
            if teval=[], the mode functions are evaluated at self.__t
        atol : float or None
            the absolute precision of the numerical intergrator, if not specified, will use self.atol
        rtol : float
            the relative precision of the numerical integrator.

        Returns
        -------
        GaugeSpec 
            the gauge-field spectrum
        """

        if t_interval is None:
            t_interval = (self.__tmin, max(self.__t))
        ks, tstart = self._find_tinit_BD(self._create_k_array(nvals, t_interval), mode="k")

        modes = np.array([self._evolve_from_BD(k, tstart[i], **SolverKwargs)
                  for i, k in enumerate(ks)])
        
        spec = GaugeSpec({"t":self.__t, "N":self.__N, "k":ks,
                    "Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        return spec
    
    def update_spectrum(self, spec : GaugeSpec, tstart, **SolverKwargs) -> GaugeSpec:
        
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
        kold, tvac = self._find_tinit_BD(spec["k"][old], mode="k")

        updatespec={"t":teval, "N":Neval, "k":kold}

        modes = []
        for i, k in enumerate(kold):
            if tvac[i] > teval[0]:
                modes.append( self._evolve_from_BD(k, tvac[i], **SolverKwargs) )
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

            vecode = np.vectorize(lambda t, y, k: self.ModeEoM(t, y, k, **self.EoMKwargs),
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
    

    def _evolve_from_BD(self, k : float, tstart : float,
                    atol : float|None=None, rtol : float=1e-5) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Evolve gauge-field modes for a fixed wavenumber in time starting from Bunch-Davies initial conditions.

        Parameters
        ----------
        k : float
           the comoving wavenumber k for which the mode function A(t,k, +/-) is evolved.
        tstart : float
            the time from which to initialise the mode evolution. 
            Should satisfy k < 10^(5/2)k_h(tstart) to ensure that the modes initialised in the Bunch-Davies vacuum
        teval : list of float
            physical time points at which the mode function A(t,k,+/-) and its derivatives will be returned
            if teval=[], the mode functions are evaluated at self.__t
        atol : float or None
            the absolute precision of the numerical intergrator, if not specified, solver will use self.atol
        rtol : float
            the relative precision of the numerical integrator.

        Returns
        -------
        yp : NDArray
            the positive helicity modes (rescaled), sqrt(2k)*A(teval, k, +)
        dyp : NDArray
            the derivative of the positive helicity modes (rescaled), sqrt(2/k)*dAdeta(teval, k, +)
        ym : NDArray
            the negative helicity modes (rescaled), sqrt(2k)*A(teval, k, -)
        dym : NDArray
            the derivative of the negative helicity modes (rescaled), sqrt(2/k)*dAdeta(teval, k, -)
        """

        #Initial conditions for y and dydt for both helicities (rescaled appropriately)
        yini = self.BDInit(tstart, k, **self.InitKwargs)

        teval = self.__t

        istart = np.where(teval>tstart)[0][0]

        yp, dyp, ym, dym = self._evolve_mode(tstart, yini, k, teval[istart:], atol, rtol)

        #conformal time needed for relative phases
        eta = self.__eta
        
        #the mode was in vacuum before tstart
        yvac = np.array([self.BDInit(t, k, **self.InitKwargs) for t in teval[:istart]]).T 
        phasevac = (np.exp(-1j*k*eta[:istart]))
        vac = yvac * phasevac

        #Create array of mode evolution stringing together vacuum and non-vacuum time evolutions to get evolution from t0 to tend
        yp = np.concatenate([(vac[0,:] + 1j*vac[2,:]), yp*np.exp(-1j*k*eta[istart])])
        dyp = np.concatenate([(vac[1,:] + 1j*vac[3,:]), dyp*np.exp(-1j*k*eta[istart])])
        ym = np.concatenate([(vac[4,:] + 1j*vac[6,:]), ym*np.exp(-1j*k*eta[istart])])
        dym = np.concatenate([(vac[5,:] + 1j*vac[7,:]), dym*np.exp(-1j*k*eta[istart])])

        return yp, dyp, ym, dym
    
    def _evolve_mode(self, tini, yini, k : float, teval : NDArray,
                    atol : float|None=None, rtol : float=1e-5):
        #Define ODE
        def ode(t, y): return self.ModeEoM(t, y, k, **self.EoMKwargs)

        if atol is None:
            atol = self.atol
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tini, max(teval)], yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)

        yp = (sol.y[0,:] + 1j*sol.y[2,:])
        dyp = (sol.y[1,:] + 1j*sol.y[3,:])
        ym = (sol.y[4,:] + 1j*sol.y[6,:])
        dym = (sol.y[5,:] + 1j*sol.y[7,:])

        return yp, dyp, ym, dym 
        

    def _create_k_array(self, nvals : int, t_interval : tuple) -> NDArray:
        """
        Create an array of wavenumbers between self.mink and self.maxk. The array is created according to the evolution of the instabiltiy scale
        such that it contains more modes close to the instability scale at late times.

        Parameters
        ----------
        nvals : int
            The size of the output array

        Returns
        -------
        NDArray
            an array of wavenumbers with size nvals
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
    
    def _find_tinit_BD(self, init : NDArray, mode : str="k") -> Tuple[NDArray, NDArray]:
        """
        Determines the solution to k = 10^(5/2)*k_h(t).
        Initial data can be given for the comoving wavenumber k, the physical time coordinates t, or e-Folds N.

        Parameters
        ----------
        init : array
           an array of physical time coordinates t, OR of e-Folds N, OR of comoving wavenumbers k.
        mode : str
            specify the content of init: "t" for physical time, "k" for comoving wavenumbers, "N" for e-Folds

        Returns
        -------
        ks : NDarray
            an array of comoving wavenumbers satisfying k=10^(5/2)k_h(tstart)
        tstart : NDarray
            an array of physical-time coordinates satisfying k=10^(5/2)k_h(tstart)

        Raises
        ------
        KeyError
            if mode is not "t", "k" or "N"
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







    
    
    
    
    
    
    
    
    