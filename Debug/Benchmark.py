import sys
import os
abspath = os.path.dirname(__file__)

from src.GEF import GEF
import numpy as np
from src.Tools.ModeByMode import ModeByMode, ReadMode

basepath = os.path.dirname(__file__)

def Benchmark(model, setting, loadGEF=True, loadspec=True):
    beta = 25
    m = 6e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    indic = {"phi":phi, "dphi":dphi}
    funcdic = {"V":V, "dV":dV}

    if model=="Classic":
        GEFPath = os.path.join(basepath, "Data/GEF+Classic_b25+m6e-6.dat")
        MbMPath = os.path.join(basepath, "Data/MbM+Classic_b25+m6e-6.dat")

    elif model=="SEOld":
        if setting["pic"]=="mixed":
            GEFPath = os.path.join(basepath, "Data/GEF+SEOld+mix_b25+m6e-6.dat")
            MbMPath = os.path.join(basepath, "Data/MbM+SEOld+mix_b25+m6e-6.dat")
        elif setting["pic"]=="electric":
            GEFPath = os.path.join(basepath, "Data/GEF+SEOld+elc_b25+m6e-6.dat")
            MbMPath = os.path.join(basepath, "Data/MbM+SEOld+elc_b25+m6e-6.dat")
        elif setting["pic"]=="magnetic":
            GEFPath = os.path.join(basepath, "Data/GEF+SEOld+mag_b25+m6e-6.dat")
            MbMPath = os.path.join(basepath, "Data/MbM+SEOld+mag_b25+m6e-6.dat")

        indic.update({"rhoChi":0., "delta":1})

    G = GEF(model, beta, indic, funcdic,
                GEFData=GEFPath, userSettings=setting)

    if loadGEF:
        G.LoadGEFData()

    if loadspec:
        spec = ReadMode(MbMPath)
    else:
        spec = None

    return G, spec