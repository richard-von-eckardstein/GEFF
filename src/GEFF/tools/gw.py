"""TODO"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

from GEFF.utility.cosmo import g_rho, g_rho_freq, g_rho_0, g_s, g_s_freq, g_s_0, T_0, M_pl, gev_to_hz, omega_r, h

from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

from numpy.typing import ArrayLike
from typing import Tuple

feq = 2.1e-17
        

class OmegaGW:
    """
    TODO
    """
    def __init__(self, values):
        #Set GEF results to Hubble units.
        values.set_units(False)

        N = values.N
        H = values.H

        Nend = N[-1]

        self.__omega = values.omega
        
        #Assess if the end of inflation is reached for this run
        """if np.log10(abs(max(N) - Nend)) > -2:
            print("This GEF run has not run reached the end of inflation. The code will assume Nend = max(N). Proceed with caution!")"""
        maxN = min(max(N), Nend)
        self.maxN = maxN

        self.__HN = CubicSpline(N, H)

        return
    
    def ktofreq(self, k : ArrayLike, Nend : float|None=None, Trh : None|float=None) -> ArrayLike:
        """
        TODO

        Input
        -----
        k : array
            an array of comoving wavenumbers k during inflation
        Nend : float or None
            the number of e-folds corresponding to the end of inflation.
            If Nend=None, it is assumed that self.maxN corresponds to the end of inflation.
        Trh : None or float
            the assumed reheating temperature in GeV. Reheating is modelled assuming an EoS parameter wrh=0.
            If Trh is None, instantaneous reheating is assumed.

        Return
        ------
        f : array
            the red-shifted frequencies in Hz
        """
        #Assumes the end of inflation is reached by the end of the run.
        if Nend is None:
            Nend = self.maxN

        Hend = self.__HN(Nend)
        if Trh is None:
            Trh = np.sqrt(3*Hend*self.__omega/np.pi)*(10/106.75)**(1/4)*M_pl
            Trh = Trh*(106.75/g_rho(Trh))**(1/4)
            Nrh = 0
        else:
            wrh = 0
            Nrh = np.log( 90*(Hend*self.__omega*M_pl**2)**2 / (np.pi**2*g_rho(Trh)*Trh**4 ) ) / ( 3 * (1 + wrh) )


        f = k*self.__omega*M_pl*gev_to_hz/(2*np.pi*np.exp(Nend)) * T_0/Trh * (g_s_0/g_s(Trh))**(1/3) * np.exp(-Nrh)

        return f
    
    def PTtoOmega(self, PT : ArrayLike, k : ArrayLike, Nend : float|None=None, Trh : None|float=None) -> Tuple[ArrayLike, ArrayLike]:
        """
        TODO

        Input
        -----
        PT : array
            an array of tensor power spectra at the end of inflation for comoving wavenumbers k
        k : array
            an array of comoving wavenumbers k during inflation
        Nend : float or None
            the number of e-folds corresponding to the end of inflation.
            If Nend=None, it is assumed that self.maxN corresponds to the end of inflation.
        Trh : None or float
            the assumed reheating temperature in GeV. Reheating is modelled assuming an EoS parameter wrh=0.
            If Trh is None, instantaneous reheating is assumed.

        Return
        ------
        f : array
            the red-shifted frequencies in Hz
        """
        
        f = self.ktofreq(k, Nend, Trh)
        if Trh is None:
            TransferRH=1
        else:
            if Nend is None:
                Nend = self.maxN
            
            frh = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/2) * (Trh/M_pl) * T_0*gev_to_hz
            fend = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/3) * (Trh/M_pl)**(1/3) * (self.__HN(Nend)*self.__omega)**(1/3) * T_0*gev_to_hz

            TransferRH = 1/(1. - 0.22*(f/frh)**1.5 + 0.65*(f/frh)**2)
            TransferRH = np.where(np.log(f/fend) < 0, TransferRH, np.zeros(f.shape))

        TransferMD = 1 + 9/32*(feq/f)**2

        h2OmegaGW = h**2*omega_r/24  * PT * (g_rho_freq(f)/g_rho_0) * (g_s_0/g_s_freq(f))**(4/3) * TransferMD * TransferRH
        
        return h2OmegaGW, f

basepath = os.path.dirname(os.path.abspath(__file__))

def IntegrateGW(f, h2OmegaGW):
    """TODO"""
    h2OmegaGW = np.where(np.log(f/1e-12) > 0, h2OmegaGW, 0)
    val = simpson(h2OmegaGW, np.log(f))
    return val

def PlotPLIS(ax : plt.Axes, names : list=[], cols : list=[], alpha : float=0.25):
    """
    Plot the sensitivity curves for current and planned gravitational wave experiments on a OmegaGW vs. frequency plot. Experiments with existing data
    are shown with filled-in sensivity curves.

    Parameters
    ----------
    ax : Axes-type
        the plot over which to overlay the sensitivity curves 
    names : list
        the names of GW experiments for which to plot the sensitivity curves. 
        Accepted names are 'LISA', 'EPTA', 'IPTA', 'HLV', 'BBO', 'HLVK', 'HLVO2', 'DECIGO', 'HL', 'NANOGrav', 'SKA', 'CE', 'ET', 'NANOGrav', 'PPTA'
        Empty lists, or lists containing invalid names are parsed to show all sensitivity curves
    cols : list
        a list of colours for which to plot the sensitivity curves. The ordering of the colors is the same as the ordering of experiments in names.
        If no colors are given (col=[]), the colors are generated at random.
    alpha : float
        the opacity with which to fill in the sensivity curves for experiments with existing data

    Returns
    -------
    ax : Axes-type
        the updated plot.
    """
    #the path to the sensitivity curve data
    path = os.path.join(basepath, "../data/plis/")
    arr = os.listdir(path)
    
    #Obtain List of experiments and running experiments
    exp = [a.replace("plis_","").replace(".dat", "") for a in arr ]
    RunningExp = ["IPTA", "NANOGrav", "PPTA", "EPTA", "HLVO2", "HLVO3"]

    #Parse Input Names
    if names!=[]:
        err=False
        for name in names:
            if name not in exp:
                print(name,"is not a recognised experiment")
                err = True
        if err:
            print("Recognised experiments are given by", exp)
            print("Defaulting to showing all known sensitivity curves")
            names=exp

    elif names==[]:
        names=exp

    arr = []
    for name in names:
        arr.append("plis_" + name + ".dat")   

    #Parse Input Colors
    if cols==[]:
        colavail = list(mcolors.CSS4_COLORS.keys())
        rndint = random.sample(range(0, len(colavail)-1), len(names))
        cols = [colavail[n] for n in rndint]

    #Create Dictionary linking experiment names to unique files and colors
    dic = {}
    for i, name in enumerate(names):
        dic[name] = {"file":arr[i],"col":cols[i],"running":name in RunningExp}


    for key in dic.keys():
        
        tab = pd.read_table(path+dic[key]["file"], comment="#", header=None).values.T
        f = 10**tab[0,:]
        SCurve = 10**tab[1,:]
        f = np.array([f[0]] + list(f) + [f[-1]])
        SCurve = np.array([1.] + list(SCurve) + [1.])
        
        if dic[key]["running"]:
            ax.fill_between(f, max(SCurve)*np.ones(f.shape), SCurve, color=dic[key]["col"], alpha=alpha)
        ax.plot(f, SCurve, color=dic[key]["col"])

    ax.set_yscale("log")
    ax.set_xscale("log")
    return ax

def ComputeSNR(fSignal, h2OmegaSignal, experiment, tobs=1.):
    """TODO"""
    path = os.path.join(basepath, "../data/strains/")
    arr = os.listdir(path)

    exp = [a.replace("strain_","").replace(".dat", "") for a in arr ]

    filename = f"strain_{experiment}.dat"

    file = path+filename
    try:
        if experiment != "NANOGrav":
            tab = pd.read_table(file, comment="#").values.T
            fNoise = 10**tab[0,:]
            h2OmegaNoise = 10**tab[1,:]
        else:
            tab = pd.read_table(file, comment="#", sep=",").values.T
            fmin = 1/(16.03*365.2425*24*3600)
            fmax = 30*fmin
            fNoise = tab[0,:]
            minind = np.where(fNoise >= fmin)[0][0]
            maxind = np.where(fNoise <= fmax)[0][-1]
            fNoise = fNoise[minind:maxind]
            h2OmegaNoise = h**2*tab[3,minind:maxind]

        content = open(file).readlines()
        if "auto-correlation" in content[1]:
            ndet = 1
        elif "cross-correlation" in content[1]: 
            ndet = 2
        else:
            raise Exception
        
    except FileNotFoundError:
        raise FileNotFoundError(f"'{experiment}' is not a recognised experiment. Recognised experiments are {exp}")
    
    indLow = np.where(fNoise[0] < fSignal)[0]
    indHigh = np.where(fNoise[-1] > fSignal)[0]
    overlap = list(set(indLow) & set(indHigh))
    overlap.sort()
    if len(overlap)==0:
        SNR = 0.
    else:
        f = fSignal[overlap]
        h2OmegaNoise = np.exp(CubicSpline(np.log(fNoise), np.log(h2OmegaNoise))(np.log(f)))
        SNR = (ndet*(tobs*365.2425*3600*24)*simpson((h2OmegaSignal[overlap]/h2OmegaNoise)**2, f))**(1/2)

    return SNR

