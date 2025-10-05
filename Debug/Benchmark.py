from GEFF import GEF, BaseGEF
import numpy as np

import os
abspath = os.path.dirname(__file__)


basepath = os.path.dirname(__file__)

def Benchmark(model, setting={}, loadGEF=True) -> BaseGEF:
    beta = 25
    m = 6e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    indic = {"phi":phi, "dphi":dphi, "V":V, "dV":dV, "beta":beta}

    if model=="classic":
        GEFPath = os.path.join(basepath, "Data/GEF+Classic_b25+m6e-6.dat")

    elif "SE" in model:
        if "noscale" in model:
            name = "SEOld"
        else:
            name = model
        if setting["pic"]=="mixed":
            GEFPath = os.path.join(basepath, f"Data/GEF+{name}+mix_b25+m6e-6.dat")
        elif setting["pic"]=="electric":
            GEFPath = os.path.join(basepath, f"Data/GEF+{name}+elc_b25+m6e-6.dat")

        elif setting["pic"]=="magnetic":
            GEFPath = os.path.join(basepath, f"Data/GEF+{name}+mag_b25+m6e-6.dat")

        indic.update({"rhoChi":0.})
    
    if model=="SE_noscale":
        indic.update({"delta":1})

    G = GEF(model, setting)(**indic)

    if loadGEF:
        G.load_GEFdata(GEFPath)

    return G