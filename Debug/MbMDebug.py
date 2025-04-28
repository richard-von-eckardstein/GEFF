from src.GEF import GEF
import numpy as np
from src.Tools.ModeByMode import ModeByMode

if __name__ == "__main__":
    #common parameters
    beta = 10
    m = 6.16e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    G = GEF("Classic", beta, {"phi":phi, "dphi":dphi}, {"V":V, "dV":dV}, GEFData="Out/GEFTesting.dat")

    G.LoadGEFData()

    MbM = ModeByMode(G, settings={})

    teval, Neval, ks, Ap, dAp, Am, dAm = MbM.ComputeSpectrum(20)

    ...

