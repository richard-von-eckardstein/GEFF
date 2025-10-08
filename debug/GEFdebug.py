from geff import make_model
import numpy as np

if __name__ == "__main__":
    model="SE_kS"
    setting={"pic":"mixed"}

    beta = 25
    m = 6e-6
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    indic = {"phi":phi, "dphi":dphi, "rhoChi":0}
    funcdic = {"V":V, "dV":dV}


    G = make_model(model, setting)({"beta":beta}, indic, funcdic)

    
    G.run(100, 120)