from geff import compile_model, BaseGEF
import numpy as np

import os

basepath = os.path.dirname(__file__)

def get_model(model, setting={}) -> BaseGEF:
    beta = 25
    m = 6e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    indic = {"phi":phi, "dphi":dphi, "V":V, "dV":dV, "beta":beta}

    if "fai" in model:
        if "basic" in model:
            indic.update({"delta":1})
        indic.update({"rhoChi":0.})

    model = compile_model(model, setting)(**indic)

    return model

def get_data(model, setting={}):
    sys = get_model(model, setting)

    if model=="pai":
        GEFPath = os.path.join(basepath, f"Data/GEF+{model}_b25+m6e-6.dat")

    elif "fai" in model:
        if setting["pic"]=="mixed":
            GEFPath = os.path.join(basepath, f"Data/GEF+{model}+mix_b25+m6e-6.dat")
        elif setting["pic"]=="electric":
            GEFPath = os.path.join(basepath, f"Data/GEF+{model}+elc_b25+m6e-6.dat")

        elif setting["pic"]=="magnetic":
            GEFPath = os.path.join(basepath, f"Data/GEF+{model}+mag_b25+m6e-6.dat")

    return sys.load_GEFdata(GEFPath)
    

