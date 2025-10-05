r"""
This module defines constants and functions which are often times encountered in cosmology. These are primarily used in `GEFF.tools.gw`
"""
import natpy as nat
import numpy as np
import pandas as pd
import os

basepath = os.path.dirname(os.path.abspath(__file__))

nat.set_active_units("HEP")

#constants
G : float = 6.67430e-11 * nat.convert(nat.m**3 * nat.kg**(-1) * nat.s**(-2), nat.GeV**(-2)) 
"""Newtons constant in GeV."""
M_pl : float = (8*np.pi*G)**(-1/2) # reduced Planck mass in GeV
"""The reduced Planck mass in GeV."""

#Conversions
gev_to_hz : float = nat.convert(nat.GeV, nat.Hz)
"""The conversion factor between Hz and GeV."""

#Planck results
T_0 : float = 2.7255 * nat.convert(nat.K, nat.GeV)
"""The temperatur of CMB photons today (in GeV)."""
h : float = 0.674
"""The reduced Hubble parameter $h$ (from [Planck 2018](https://arxiv.org/abs/1807.06209))."""
H0 : float = h*100*nat.convert(nat.km * nat.s**(-1) * nat.Mpc**(-1), nat.GeV)
"""The Hubble rate today (in GeV)."""

gtab = pd.read_table(os.path.join(basepath, "../data/eff_dof.dat"), sep=" ", comment="#").values.T
gdic = {"T":gtab[0,:],"f":gtab[1,:],"grho":gtab[2,:], "gs":gtab[3,:]}

def g_rho(T : float) -> float:
    """
    Effective number of relativistic degrees of freedom as a function of temperature, $g_{*}(T)$.

    Data is taken from [2005.03544](https://arxiv.org/abs/2005.03544).
    """
    return np.interp(T, gdic["T"], gdic["grho"])

def g_s(T : float) -> float:
    """
    Effective number of entropic degrees of freedom as a function of temperature, $g_{*,S}(T)$.

    Data is taken from [2005.03544](https://arxiv.org/abs/2005.03544).
    """
    return np.interp(T, gdic["T"], gdic["gs"])

def g_rho_freq(f : float) -> float:
    """
    Effective number of relativistic degrees of freedom as a function of frequency, $g_{*}(T(f))$.

    Data is taken from [2005.03544](https://arxiv.org/abs/2005.03544).
    """
    return np.interp(f, gdic["f"], gdic["grho"])

def g_s_freq(f : float) -> float:
    """
    Effective number of entropic degrees of freedom as a function of frequency, $g_{*,S}(T(f))$.

    Data is taken from [2005.03544](https://arxiv.org/abs/2005.03544).
    """
    return np.interp(f, gdic["f"], gdic["gs"])



g_rho_0 : float = g_rho(T_0)
"""The effective number of relativistic degrees of freedom today, $g_{*}(T_0)$."""
g_s_0 : float = g_s(T_0)
"""The effective number of entropic degrees of freedom today, $g_{*,S}(T_0)$."""
omega_r : float = np.pi**2*g_rho_0/(90*M_pl**2*H0**2)*T_0**4
"""The density parameter for radiation today, $\Omega_r$."""

feq = 2.1e-17
"""Frequency corresponding to mattter-radiation equality"""

