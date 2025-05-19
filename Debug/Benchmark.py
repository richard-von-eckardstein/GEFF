import sys
import os
abspath = os.path.dirname(__file__)

from src.GEF import GEF
import numpy as np
from src.Tools.ModeByMode import ModeByMode, ReadMode

basepath = os.path.dirname(__file__)

def Benchmark(model, setting):
    beta = 25
    m = 6e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    indic = {"phi":phi, "dphi":dphi}
    funcdic = {"V":V, "dV":dV}

    if model=="Classic":
        GEFPath = os.path.join(basepath, "Data/GEF+Classic_b25+m1e-6.dat")
        MbMPath = os.path.join(basepath, "Data/MbM+Classic_b25+m1e-6.dat")

    elif model=="SEOld":
        GEFPath="Out/Schwinger_OldMix_b25_m-6.dat"
        indic.update({"rhoChi":0., "delta":1})

    G = GEF(model, beta, indic, funcdic,
                GEFData=GEFPath)

    G.LoadGEFData()

    MbM = G.MbM(G)

    spec = ReadMode(MbMPath)

    return G, spec