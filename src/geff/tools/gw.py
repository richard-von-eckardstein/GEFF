import numpy as np

from geff._docs.docs_gw import DOCS
from geff.utility.cosmo import g_rho, g_rho_freq, g_rho_0, g_s, g_s_freq, g_s_0, T_0, M_pl, gev_to_hz, omega_r, h, feq

from numpy.typing import ArrayLike
from typing import Tuple

__doc__ = DOCS["module"]
        
def omega_gw(k:np.ndarray, PT:np.ndarray, Nend:float, Hend:float, Trh:None|float=None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute $h^2 \Omega_{\rm GW}(f)$ from a tensor power spectrum.

    Parameters
    ----------
    k : NDarray
        momenta in Planck units
    PT : NDarray
        the tensor power spectrum at the end of inflation as a function of momentum
    Nend : float
        the number of e-folds at the end of inflation
    Hend : float
        the Hubble rate at the end of inflation (in Planck units)
    Trh : None or float
        the reheating temperature in GeV. If None, instantaneous reheating is assumed.

    Returns
    -------
    f : NDarray
        frequencies today (in Hz)
    h2OmegaGw : NDarray
        the gravitational-wave spectrum as a function of frequency today
    """

    f = k_to_f(k, Nend, Hend, Trh)
    if Trh is None:
        TransferRH=1
    else:
        frh = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/2) * (Trh/M_pl) * T_0*gev_to_hz
        fend = 1/(2*np.pi) * (g_s_0/g_s(Trh))**(1/3) * (np.pi**2*g_rho(Trh)/90)**(1/3) * (Trh/M_pl)**(1/3) * (Hend)**(1/3) * T_0*gev_to_hz

        TransferRH = 1/(1. - 0.22*(f/frh)**1.5 + 0.65*(f/frh)**2)
        TransferRH = np.where(np.log(f/fend) < 0, TransferRH, np.zeros(f.shape))

    TransferMD = 1 + 9/32*(feq/f)**2

    h2OmegaGW = h**2*omega_r/24  * PT * (g_rho_freq(f)/g_rho_0) * (g_s_0/g_s_freq(f))**(4/3) * TransferMD * TransferRH

    return f, h2OmegaGW

def k_to_f(k:np.ndarray, Nend:float, Hend:float, Trh:None|float=None) -> ArrayLike:
    r"""
    Compute frequency today from momentum at the end of inflation

    Parameters
    ----------
    k : NDarray
        momenta in Planck units
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
