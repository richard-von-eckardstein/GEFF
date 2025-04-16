from src.GEF import GEF
import numpy as np

if __name__ == "__main__":
    #common parameters
    beta = 30
    m = 6.16e-8
    phi = 15.55
    dphi = -np.sqrt(2/3)*m
    V = lambda x: 1/2*m**2*x**2
    dV = lambda x: m**2*x

    ntr = 150
    tend = 120
    atol = 1e-20
    rtol = 1e-6
    reachNend = True

    G = GEF("Classic", beta, {"phi":phi, "dphi":dphi}, {"V":V, "dV":dV},
             GEFData="Out/Testing.dat")
    sol = G.Solver.RunGEF(ntr, 60, atol=atol, rtol=rtol, nmodes=500, printstats=False,
                           ensureConvergence=False, reachNend=reachNend, maxattempts=3)

    G.Solver.ParseArrToUnitSystem(sol.t, sol.y, G)

