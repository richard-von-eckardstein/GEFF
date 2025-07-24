import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import solve_ivp
from scipy.integrate import quad, simps

from src.BGQuantities.BGTypes import Val, BGSystem
from src.EoMsANDFunctions.ModeEoMs import ModeEoMClassic, BDClassic

from numpy.typing import NDArray
from typing import Tuple, Callable

class GaugeSpec(dict):
    """
    A class representing a spectrum of gauge-field modes as a function of time.
    This class inherits from 'dict', and necessarily needs the following keys:
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
        arrays of shape (len(k), len(t)) containing the mode functions sqrt(2*k)*A_\pm(k, t)
    dAp, dAm : NDarray
        arrays of shape (len(k), len(t)) containing the mode-function derivatives  sqrt(2/k)*e^N*dA_\pm(k, t)/dt

    Class Methods
    -------------
    ReadSpec
        Initialise the class from data stored in a file.

    Methods
    -------
    SaveSpec()
        Store the spectrum in a file.
    GetDim()
        Retrieve the number of modes and times encoded in the spectrum 
    TSlice()
        Retrieve the spectrum at a moment of time
    KSlice()
        Retrieve the spectrums evolution for a fixed wavenumber

    Example
    -------
    >>> spec = GaugeSpec.ReadSpec(somefile)
    >>> specslice = spec.TSlice(100) #return the spectrum at spec["t"][100]
    >>> print(f"This is the spectrum of positive-helicity modes at time {specslice['t']}:)
    >>> print(specslice["Ap"]})
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
    def ReadSpec(cls, path : str):
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
        return cls(ReadMode(path))
    
    def SaveSpec(self, path : str, thinning = 5):
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
            specslice = self.KSlice(j)
            logk = np.log10(specslice["k"])
            for key in ["Ap", "dAp", "Am", "dAm"]:
                dictmp = {key + "_" + str(j) :np.array([logk] + 
                                                       list(specslice[key][-1::-thinning][::-1]))}
                dic.update(dictmp)
    
        DirName = os.getcwd()
    
        path = os.path.join(DirName, path)
    
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)
        
        return

    def GetDim(self) -> dict:
        """
        Retrieve the spectrums evolution for a fixed wavenumber

        Returns
        -------
        dict
            a dictionary encoding the number of modes 
            and the number of times stored in the spectrum
        """
        return {"kdim":len(self["k"]), "tdim":len(self["t"])}
    
    def TSlice(self, ind : int) -> dict:
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

        specslice = {}
        for key, item in self.items():
            if key in ["N", "t", "cut"]:
                specslice[key] = self[key][ind]
            elif key=="k":
                specslice[key] = self[key]
            else:
                specslice[key] = self[key][:,ind]
        return GaugeSpecSlice(specslice)
    
    def KSlice(self, ind : int) -> dict:
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

        specslice = {}
        for key, item in self.items():
            if key in ["N", "t", "cut"]:
                specslice[key] = self[key]
            elif key=="k":
                specslice[key] = self[key][ind]
            else:
                specslice[key] = self[key][ind,:]
        return specslice
    
    def MergeSpectra(self, spec):
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
    
    def AddMomenta(self, spec):
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
    
    def RemoveMomenta(self, ind):
        self["k"] = np.delete(self["k"], ind)
        for md in ["Ap", "dAp", "Am", "dAm"]:
            self[md] = np.delete(self[md], ind, axis=0)
        return
    
    def CheckOverlap(self, t):
        mask = np.isin(t, self["t"], assume_unique=True)
        if len(t[mask]) != len(self["t"]):
            print("The times in the current GaugeSpec instance are " \
            "not a subset of the times in the BGSystem.")
            print("Reverting to interpolation.")
            return False, None
        else:
            return True, mask
    
    def AddCutOff(self, BG : BGSystem, cutoff="kh"):
        units = BG.GetUnits()
        BG.SetUnits(False)

        scale = getattr(BG, cutoff)

        bl, mask = self.CheckOverlap(BG.t)

        if bl:
            self["cut"] = scale[mask]
        else:
            self["cut"] = CubicSpline(BG.t, scale)(self["t"])
        
        BG.SetUnits(units)

        return self["cut"]
    
    def GetReferenceGaugeFields(self, BG : BGSystem, references=["E", "B", "G"], cutoff="kh"): 
        units = BG.GetUnits()
        BG.SetUnits(False)

        scale = getattr(BG, cutoff)

        bl, mask = self.CheckOverlap(BG.t)

        Fref = []
        for val in references: 
            val_arr = (getattr(BG, val)*(BG.a/scale)**4)
            if bl:
                Fref.append( val_arr[mask] )
            else:
                Fref.append( CubicSpline(BG.t, val_arr)(self["t"]) ) 

        BG.SetUnits(units)

        return Fref
    
    def IntegrateSpec(self, BG : BGSystem, n : int=0, cutoff="kh", **IntegratorKwargs) -> NDArray:
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

        self.AddCutOff(BG, cutoff)

        tdim = self.GetDim()["tdim"]

        FMbM = np.zeros((tdim, 3,2))
        for i in range(tdim):
            specslice = self.TSlice(i)
            FMbM[i,:] = specslice.IntegrateSpecSlice(n=n, **IntegratorKwargs)
        
        return FMbM
    
    def EstimateGEFError(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                         **IntegratorKwargs):
        FMbM = self.IntegrateSpec(BG, n=0, cutoff=cutoff, **IntegratorKwargs)
        Fref = self.GetReferenceGaugeFields(BG, references, cutoff)

        errs = []

        for i, spl in enumerate(Fref):
            err =  abs( 1 - FMbM[:,i,0]/ spl )
            errs.append(np.where(np.isnan(err), 1.0, err))

        return errs
    
    def ProcessError(self, errs, binning, errthr):
        teval = self["t"]

        terr = teval[-1::-binning][::-1]
        tbins = terr[1:] + (terr[1:] - terr[:-1])/2
        count, _  = np.histogram(teval, bins=tbins)

        bin_errs = []
        for err in errs:
            sum, _  = np.histogram(teval, bins=tbins, weights=err)
            bin_errs.append(sum/count)
        terr = terr[2:] 

        removals = []
        for err in bin_errs:
            #remove the first few errors where the density of modes is low:
            removals.append(np.where(err < errthr)[0][0])
        #ind = 0
        ind = max(removals)
        bin_errs = [err[ind:] for err in bin_errs]
        terr = terr[ind:]

        return bin_errs, terr
    
    @staticmethod
    def ErrorSummary(errs, terr, references):
        print("The mode-by-mode comparison finds the following relative deviations from the GEF solution:")
        for i, key in enumerate(references):
            err = errs[i]
            errind = np.where(err == max(err))
            rmserr = np.round(100*np.sqrt(np.sum(err**2)/len(err)), 1)
            maxerr = np.round(100*err[errind][0], 1)
            tmaxerr = terr[errind][0]#np.round(Nerr[errind][0], 1)
            errend = np.round(100*err[-1], 1)
            terrend = terr[-1]#np.round(Nerr[-1], 1)
            print(f"-- {key} --")
            print(f"maximum relative deviation: {maxerr}% at t={tmaxerr}")
            print(f"final relative deviation: {errend}% at t={terrend}")
            print(f"RMS relative deviation: {rmserr}% at t={terrend}")
        return


    def CompareToBackgroundSolution(self, BG : BGSystem, references : list[str]=["E", "B", "G"], cutoff : str="kh",
                                    errthr=0.025, steps=5, verbose : bool=True,
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

        errs = self.EstimateGEFError(BG, references, cutoff, **IntegratorKwargs)

        bin_errs, bin_terr = self.ProcessError(errs, steps, errthr)

        if verbose:
            self.ErrorSummary(bin_errs, bin_terr, references)

        return bin_errs, bin_terr, errs
    
class GaugeSpecSlice(dict):
    def __init__(self, modedic):
        super().__init__(modedic)

    def ESpec(self, lam):
        return abs(self["dA"+lam])**2
    
    def BSpec(self, lam):
        return abs(self["A"+lam])**2
    
    def GSpec(self, lam):
        return (self["A"+lam].conjugate()*self["dA"+lam]).real
    
    def SimpsInt(self, integrand, x):
        integrand = integrand*np.exp(x)
        return simps(integrand, x)

    def QuadInt(self, integrand, x, epsabs : float=1e-20, epsrel : float=1e-4):
        msk = np.where(abs(integrand) > epsrel*1e-2*abs(integrand))[0]
        if len(msk) > 0:
            spl = CubicSpline(x, np.log(abs(integrand)+epsabs/10))
            sgn = CubicSpline(x, np.sign(integrand))
            f = lambda x: sgn(x)*np.exp(spl(x) + x)
            val, err = quad(f, x[msk][0], 0., epsabs=epsabs, epsrel=epsrel)
        else:
            return 0
        return np.array([val, err])
    
    def OldQuadInt(self, integrand, x, epsabs : float=1e-20, epsrel : float=1e-4):
        spl = CubicSpline(x, integrand)
        f = lambda x: spl(x)*np.exp(x)
        val, err = quad(f, -200, 0., epsabs=epsabs, epsrel=epsrel)
        return np.array([val, err])
        
    def IntegrateSpecSlice(self, n : int=0, epsabs : float=1e-20, epsrel : float=1e-4,
                            method="simpson", modethr=100) -> Tuple[NDArray, NDArray]:
        """
        Integrate an input spectrum at a fixed time t to obtain (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)

        Parameters
        ----------
        specAtT : dict
            the spectrum at time t, obtained by GaugeSpec.TSlice
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
            integs[0,i,:] = self.ESpec(lam)
            integs[1,i,:] = self.BSpec(lam)
            integs[2,i,:] = sgn*self.GSpec(lam)

        res = np.zeros((3,2))
        
        for i in range(3):
            if method=="simpson":
                msk = np.where(x < 0)[0]
                if len(msk) < modethr: #cannot trust simpsons integration for too few modes.
                    return res
                x = x[msk]
                res[i,0] = (self.SimpsInt( integs[i,0,msk] ,x) 
                                         + (-1)**n*self.SimpsInt(integs[i,1,msk], x) )
                res[i,1] = 1e-6*res[i,0]

            elif method=="quad":
                resp = self.QuadInt( integs[i,0,:], x, epsabs, epsrel)
                resm = self.QuadInt(  (-1)**n*integs[i,1,:], x, epsabs, epsrel)
                res[i,:] = resp +resm

            elif method=="old":
                res[i,:] = self.OldQuadInt( integs[i,0,:] + (-1)**n*integs[i,1,:], x, epsabs, epsrel)
            

        res = 1/(2*np.pi)**2*res/(n+4)

        res[:,1] = abs(res[:,1]/res[:,0])
        return res
    
def ReadMode(path : str) -> GaugeSpec:   
    """
    Load a gauge-field spectrum from a file.

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

    spec = GaugeSpec({"t":t, "N":N, "k":k,
                    "Ap":Ap, "dAp":dAp, "Am":Am, "dAm":dAm})
    
    return spec
    
def ModeSolver(ModeEq : Callable, EoMkeys : list, BDEq : Callable, Initkeys : list, default_atol : float=1e-3):
    """
    Class-factory creating a custom ModeByMode-class with new mode equations and initial conditions adapted to a modified version of the GEF.

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
    
    class ModeSolver(ModeByMode):
        """
        A custom ModeByMode-class with new mode equations and initial conditions adapted to a modified version of the GEF
        It Inherits all methods from ModeByMode but overwrites the following class attributes
            - ModeEoM
            - EoMKwargs
            - BDInit
            - InitKwargs
            - default-atol
        This entails that 'ComputeModeSpectrum' will now evolve modes according to BDInit and ModeEom.

        Methods
        -------
        ComputeModeSpectrum()
            Compute a gauge-field spectrum by evolving each mode in time starting from Bunch-Davies initial conditions
        IntegrateSpec()
            Integrate an input spectrum to determine the expectation values of (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)
        CompareToBackgroundSolution()
            Estimate the relative deviation in E^2, B^2, E.B between a GEF solution and a mode-spetrum as a function of e-folds

        Example
        -------
        >>> M = ModeSolver(G) #initialise the class by a BGSystem or GEF instance
        ... 
        >>> spec = M.ComputeModeSpectrum(500) #compute a gauge-field spectrum of 500 modes from G
        >>> errs, Nerr = M.CompareToBackgroundSolution(spec) #asses the agreement between G and spec
        """
        
        #Overwrite class attibutes of ModeByMode with new mode equations, boundary conditions and default tolerances.
        ModeEoM = staticmethod(ModeEq)
        EoMKwargs = dict(zip(EoMkeys, [None for x in EoMkeys]))

        BDInit = staticmethod(BDEq)
        InitKwargs = dict(zip(Initkeys, [None for x in Initkeys]))

        atol=default_atol
        
        def __init__(self, values):
            super().__init__(values)
    
    return ModeSolver


class ModeByMode:
    """
    A class used to solve the gauge-field mode equations for standard axion inflation based on a solution to the GEF equations.

    Methods
    -------
    ComputeModeSpectrum()
        Compute a gauge-field spectrum by evolving each mode in time starting from Bunch-Davies initial conditions
    IntegrateSpec()
        Integrate an input spectrum to determine the expectation values of (E, rot^n E), (B, rot^n B), (E, rot^n B), rescaled by (kh/a)^(n+4)
    CompareToBackgroundSolution()
        Estimate the relative deviation in E^2, B^2, E.B between a GEF solution and a mode-spetrum as a function of e-folds

    Example
    -------
    >>> M = ModeByMode(G) #initialise the class by a BGSystem or GEF instance
    ... 
    >>> spec = M.ComputeModeSpectrum(500) #compute a gauge-field spectrum of 500 modes from G
    >>> errs, Nerr = M.CompareToBackgroundSolution(spec) #asses the agreement between G and spec
    """

    #Class to compute the gauge-field mode time evolution and the E2, B2, EB quantum expectation values from the modes
    ModeEoM = staticmethod(ModeEoMClassic)
    EoMKwargs = {"a":None, "H":None, "xi":None}
    BDInit = staticmethod(BDClassic)
    InitKwargs = {}
    atol=1e-3

    def __init__(self, values : BGSystem):
        #Ensure that all values from the GEF are imported without units
        values.SetUnits(False)

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
        deta = lambda t, y: 1/self.__af(t)
        
        soleta = solve_ivp(deta, [min(self.__t), max(self.__t)], np.array([-1]), t_eval=self.__t)

        self.__eta = soleta.y[0,:]

        #Nend = G.EndOfInflation()

        maxN = max(self.__N)#min(max(self.__N), Nend)
        
        #Define suitable range of wavenumbers which can be considered given the background dynamics. mink might still change
        self.maxk = CubicSpline(self.__N, 10*kh)(maxN)
        self.mink = 10**4*kh[0]

        #find lowest t value corresponding to kh(t) = 10*mink
        self.__tmin = self.__t[np.where(kh >= self.mink)][0]
        
        return
    
    def ComputeModeSpectrum(self, nvals : int, t_interval=None, **SolverKwargs) -> GaugeSpec:
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

        if t_interval==None:
            t_interval = (self.__tmin, max(self.__t))
        ks, tstart = self.InitialKTN(self.WavenumberArray(nvals, t_interval), mode="k")

        modes = np.array([self.EvolveFromBD(k, tstart[i], **SolverKwargs)
                  for i, k in enumerate(ks)])
        
        spec = GaugeSpec({"t":self.__t, "N":self.__N, "k":ks,
                    "Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        return spec
    
    #at the moment, it does not seem feasable to use this.
    def UpdateSpectrum(self, spec : GaugeSpec, tstart, **SolverKwargs) -> GaugeSpec:
        
        indstart = np.where(tstart <= self.__t)[0][0]
        teval = self.__t[indstart:]
        Neval = self.__N[indstart:]
        
        tend = teval[-1]
        indstart = np.where(spec["t"]<teval[0])[0][-1]
        startspec = spec.TSlice(indstart)
        tstart = startspec["t"]

        #keep mode-evolution from old spectrum for modes with k < 10*kh(tstart)
        old = np.where(spec["k"] < 10*self.__khf(tstart))[0]
        new = np.where(spec["k"] > 10*self.__khf(tstart))[0]

        #Remove modes which need to be renewed from old spectrum:
        spec.RemoveMomenta(new)

        #add new modes, adjusting for longer time-span:
        #rethink this! It seems like overkill
        n_newmodes = int(len(new)*max((teval[-1] - teval[0])/(max(spec["t"]) - tstart), 1))

        #Update evolution of modes in spec:
        kold, tvac = self.InitialKTN(spec["k"][old], mode="k")

        updatespec={"t":teval, "N":Neval, "k":kold}

        modes = []
        for i, k in enumerate(kold):
            if tvac[i] > teval[0]:
                modes.append( self.EvolveFromBD(k, tvac[i], **SolverKwargs) )
            else:
                yini = np.array(
                            [startspec["Ap"][i].real, startspec["dAp"][i].real,
                            startspec["Ap"][i].imag, startspec["dAp"][i].imag,
                            startspec["Am"][i].real, startspec["dAm"][i].real,
                            startspec["Am"][i].imag, startspec["dAm"][i].imag]
                            )
                
                modes.append( self.EvolveMode(tstart, yini, k, teval, **SolverKwargs) ) 
        
        modes = np.array(modes)

        updatespec.update({"Ap":modes[:,0,:], "dAp":modes[:,1,:], "Am":modes[:,2,:], "dAm":modes[:,3,:]})

        spec.MergeSpectra(GaugeSpec(updatespec))

        if n_newmodes > 0:
            newspec = self.ComputeModeSpectrum(n_newmodes, t_interval=(tstart, tend))
            #Add new modes
            spec.AddMomenta(newspec)
        
        return spec
    
    """def EvolveSpectrum(self, spec : GaugeSpec, Nstart, Nstep : float=0.1, atol : float|None=None, rtol : float=1e-5) -> GaugeSpec:
        Neval = np.arange(Nstart, max(self.__N), Nstep)
        teval = CubicSpline(self.__N, self.__t)(Neval)

        indstart = np.where(spec["N"]<Nstart)[0][-1]
        startspec = spec.TSlice(indstart)

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
        
        if atol==None:
            atol = self.atol
        
        yini = np.dstack( (startspec["Ap"].real, startspec["dAp"].real,
                            startspec["Ap"].imag, startspec["dAp"].imag,
                            startspec["Am"].real, startspec["dAm"].real,
                            startspec["Am"].imag, startspec["dAm"].imag))[0].reshape(8*klen)
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [startspec["t"], max(teval)],
                         yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)

        newspec = {"t":teval, "N":Neval, "k":spec["k"]}
        for i, key in enumerate(["Ap", "dAp"]):
            newspec[key] = sol.y[i::8,:] + 1j*sol.y[2+i::8,:]
        for i, key in enumerate(["Am", "dAm"]):
            newspec[key] = sol.y[4+i::8,:] + 1j*sol.y[6+i::8,:]
            
        spec.MergeSpectra(GaugeSpec(newspec))

        return spec"""
    
    def EvolveFromBD(self, k : float, tstart : float,
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

        yp, dyp, ym, dym = self.EvolveMode(tstart, yini, k, teval[istart:], atol, rtol)

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
    
    def EvolveMode(self, tini, yini, k : float, teval : NDArray,
                    atol : float|None=None, rtol : float=1e-5):
        #Define ODE
        ode = lambda t, y: self.ModeEoM(t, y, k, **self.EoMKwargs)

        if atol==None:
            atol = self.atol
        
        #Solve differential equation from tstart to tmax
        sol = solve_ivp(ode, [tini, max(teval)], yini, t_eval=teval, method="RK45", atol=atol, rtol=rtol)

        yp = (sol.y[0,:] + 1j*sol.y[2,:])
        dyp = (sol.y[1,:] + 1j*sol.y[3,:])
        ym = (sol.y[4,:] + 1j*sol.y[6,:])
        dym = (sol.y[5,:] + 1j*sol.y[7,:])

        return yp, dyp, ym, dym 
        

    def WavenumberArray(self, nvals : int, t_interval : tuple) -> NDArray:
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
            numnewvals = nvals - len(logks)
            if numnewvals >= len(logks):
                newvals = (logks[1:] + logks[:-1])/2
            else:
                newvals = (logks[-numnewvals:] + logks[-numnewvals-1:-1])/2
            logks = np.sort(np.concatenate([logks, newvals]))
        return np.exp(logks)
    
    def InitialKTN(self, init : NDArray, mode : str="k") -> Tuple[NDArray, NDArray]:
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
        K : NDarray
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
            k = 10**(5/2)*self.__khf(tstart)

        elif mode=="k":
            k = init
            
            tstart = []
            for i, l in enumerate(k):
                ttmp  = self.__t[np.where(l >= 10**(5/2)*self.__khf(self.__t))[0][-1]]
                tstart.append(ttmp)
            tstart = np.array(tstart)

        elif mode=="N":
            tstart = CubicSpline(self.__N, self.__t)(init)
            k = 10**(5/2)*self.__khf(tstart)

        else:
            raise KeyError("'mode' must be 't', 'k' or 'N'")

        return k, tstart







    
    
    
    
    
    
    
    
    