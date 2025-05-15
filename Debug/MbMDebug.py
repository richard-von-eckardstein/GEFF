import sys
import os
abspath = os.path.dirname(__file__)
path = os.path.join(abspath, "../")
sys.path.append(path)


from src.GEF import GEF
import numpy as np
from src.Tools.ModeByMode import ModeByMode

if __name__ == "__main__":
    #common parameters
    beta = 25
    m = 6.16e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    G = GEF("SEOld", beta, {"phi":phi, "dphi":dphi, "rhoChi":0., "delta":1}, {"V":V, "dV":dV},
            GEFData="Out/Schwinger_OldMix_b25_m-6.dat")#, GEFData="Out/Test.dat")

    G.LoadGEFData()

    MbM = G.MbM(G)

    spec = MbM.ComputeModeSpectrum(20)

    print(spec.GetDim())

