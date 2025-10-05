import numpy as np
import pandas as pd
import os

from GEFF._docs.docs_gw import DOCS
from GEFF.utility.cosmo import g_rho, g_rho_freq, g_rho_0, g_s, g_s_freq, g_s_0, T_0, M_pl, gev_to_hz, omega_r, h, feq
import GEFF

from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

from numpy.typing import ArrayLike
from typing import Tuple
from tabulate import tabulate

__doc__ = DOCS["module"]

basepath = os.path.join(os.path.dirname(os.path.abspath(GEFF.__file__)), "data/")
        
def omega_gw(PT:np.ndarray, k:np.ndarray,  Nend:float, Hend:float, Trh:None|float=None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute $h^2 \Omega_{\rm GW}(f)$ from a tensor power spectrum.

    Parameters
    ----------
    PT : NDarray
        the tensor power spectrum at the end of inflation as a function of momentum
    k : NDarray
        momenta for pt, in Planck units
    Nend : float
        the number of e-folds at the end of inflation
    Hend : float
        the Hubble rate at the end of inflation
    Trh : None or float
        the reheating temperature in GeV. If None, instantaneous reheating is assumed.

    Returns
    -------
    h2OmegaGw : NDarray
        the gravitational-wave spectrum as a function of frequency today
    f : NDarray
        frequencies today (in Hz)
    """

    f = k_to_f(k, Nend, Trh)
    if Trh is None:
        TransferRH=1
        
        frh = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/2) * (Trh/M_pl) * T_0*gev_to_hz
        fend = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/3) * (Trh/M_pl)**(1/3) * (Hend)**(1/3) * T_0*gev_to_hz

        TransferRH = 1/(1. - 0.22*(f/frh)**1.5 + 0.65*(f/frh)**2)
        TransferRH = np.where(np.log(f/fend) < 0, TransferRH, np.zeros(f.shape))

    TransferMD = 1 + 9/32*(feq/f)**2

    h2OmegaGW = h**2*omega_r/24  * PT * (g_rho_freq(f)/g_rho_0) * (g_s_0/g_s_freq(f))**(4/3) * TransferMD * TransferRH

    return h2OmegaGW, f

def k_to_f(k:np.ndarray, Nend:float, Hend:float, Trh:None|float=None) -> ArrayLike:
    r"""
    Compute frequency today from momentum at the end of inflation

    Parameters
    ----------
    k : NDarray
        momenta for pt, in Planck units
    Nend : float
        the number of e-folds at the end of inflation
    Hend : float
        the Hubble rate at the end of inflation
    Trh : None or float
        the reheating temperature in GeV. If None, instantaneous reheating is assumed.

    Return
    ------
    f : NDarray
        frequencies today (in Hz)
    """

    if Trh is None:
        Trh = np.sqrt(3*Hend/np.pi)*(10/106.75)**(1/4)*M_pl
        Trh = Trh*(106.75/g_rho(Trh))**(1/4)
        Nrh = 0
    else:
        wrh = 0
        Nrh = np.log( 90*(Hend*M_pl**2)**2 / (np.pi**2*g_rho(Trh)*Trh**4 ) ) / ( 3 * (1 + wrh) )


    f = k*M_pl*gev_to_hz/(2*np.pi*np.exp(Nend)) * T_0/Trh * (g_s_0/g_s(Trh))**(1/3) * np.exp(-Nrh)

    return f

def integrate_gw(h2OmegaGW:np.ndarray, f:np.ndarray):
    r"""
    Integrate a gravitational wave spectrum over $\log f$ from $f_{\rm BBN} = 10^{-12} \, Hz$ to $f_{\rm max}$

    Parameters
    ----------
    h2OmegaGW : NDarray
        gravitational wave spectrum ($h^2 \Omega_{\rm GW}$)
    f : NDarray
        frequencies in Hz

    Returns 
    -------
    integral : float
        the value of the integral

    """
    h2OmegaGW = np.where(np.log(f/1e-12) > 0, h2OmegaGW, 0)
    integral = simpson(h2OmegaGW, np.log(f))
    return integral

def known_gw_obs(verbose:bool=False):
    """
    Get a dictionary of all recognized GW experiments.


    Parameters
    ----------
    verbose : boolean
        If True, the function prints the dictionary content as a table.

    Returns
    -------
    observatories : dict
        Output dictionary denoting for each experiments, 
        if data for strain noise spectra, power-law indicated sensitivity (PLIS) curves, or both, are present.
    """

    plis_path = os.path.join(basepath, "plis/")
    plis_names = [a.replace("plis_", "").replace(".dat", "") for a in os.listdir(plis_path)]

    strain_path = os.path.join(basepath, "strains/")
    strain_names = [a.replace("strain_", "").replace(".dat", "") for a in os.listdir(strain_path)]

    names = list( set(plis_names).union(set(strain_names)) )
    names.sort()

    observatories = {name : {"PLIS":name in plis_names, "strain":name in strain_names} for name in names}

    if verbose:
        content = [[observatories[key]["PLIS"], observatories[key]["strain"]] for key in names]
        print("The following GW observatories are known:")
        print(tabulate(content, headers=["Name", "PLIS", "Strain"], showindex=observatories.keys(), tablefmt="simple")+"\n")

    return observatories

def get_plis(experiment:str):
    """
    Retrieve the power-law integrate sensitivity curves (PLIS) for a given GW observatory. 
    
    To get a list of known experiments, use `known_gw_obs`.

    Parameters
    ----------
    experiment : str
        the name of the experiment

    Returns
    -------
    SCurve : NDarray
        PLIS as function of frequency
    f : NDarray
        frequencies in Hz
    """
    expdic = known_gw_obs()

    try:
        assert experiment in expdic.keys()
        assert expdic[experiment]["PLIS"]
    except AssertionError:
        raise KeyError(f"No PLIS for '{experiment}'.")
    else:
        file  = os.path.join(basepath, f"plis/plis_{experiment}.dat")
        tab = pd.read_table(file, comment="#", header=None).values.T
        f = 10**tab[0,:]
        SCurve = 10**tab[1,:]
        f = np.array([f[0]] + list(f) + [f[-1]])
        SCurve= np.array([1.] + list(SCurve) + [1.])

    return SCurve, f

def compute_SNR(h2Omega:np.ndarray, f:np.ndarray, experiment:str, tobs:float=1.):
    """"
    Estimate the signal-to-noise ratio for a GW spectrum and a given GW observatory.
    
    To get a list of known experiments, use `known_gw_obs`.

    Parameters
    ----------
    h2Omega : NDarray
        GW spectrum as function of frequency
    f : NDarray
        frequency in Hz
    experiment : str
        the name of the experiment
    tobs : float
        observation time in years.

    Returns
    -------
    SNR : float
        the estimated signal-to-noise ratio.
    """
    expdic = known_gw_obs()

    try:
        assert experiment in expdic.keys()
        assert expdic[experiment]["strain"]
    except AssertionError:
        raise KeyError(f"No strain noise spectra for '{experiment}'.")
    else:
        file  = os.path.join(basepath, f"plis/strain_{experiment}.dat")
        if experiment != "NANOGrav":
            tab = pd.read_table(file, comment="#").values.T
            fNoise = 10**tab[0,:]
            h2OmegaNoise = 10**tab[1,:]
        else:
            tab = pd.read_table(file, comment="#", sep=",").values.T
            fmin = 1/(16.03*365.2425*24*3600)
            fmax = 30*fmin
            fNoise = tab[0,:]
            minind = np.searchsorted(fNoise, fmin, "right")
            maxind = np.searchsorted(fNoise, fmax, "left")
            fNoise = fNoise[minind:maxind]
            h2OmegaNoise = h**2*tab[3,minind:maxind]

        content = open(file).readlines()
        if "auto-correlation" in content[1]:
            ndet = 1
        elif "cross-correlation" in content[1]: 
            ndet = 2
    
    indLow = np.where(fNoise[0] < f)[0]
    indHigh = np.where(fNoise[-1] > f)[0]
    overlap = list(set(indLow) & set(indHigh))
    overlap.sort()
    if len(overlap)==0:
        SNR = 0.
    else:
        f = f[overlap]
        h2OmegaNoise = np.exp(CubicSpline(np.log(fNoise), np.log(h2OmegaNoise))(np.log(f)))
        SNR = (ndet*(tobs*365.2425*3600*24)*simpson((h2Omega[overlap]/h2OmegaNoise)**2, f))**(1/2)

    return SNR

