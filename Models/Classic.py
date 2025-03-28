import numpy as np
from src.SolverFunctions.ClassicEoMs import *
from src.SolverFunctions.WhittakerFuncs import WhittakerApprox, WhittakerExact
from src.SolverFunctions.Utility import Heaviside

name = "Classic"

modelQuantities = {}
modelFunctions = {}
modelRhos = []

modelSettings = {}

def Initialise(GEF, ntr):
    yini = np.zeros((ntr+1)*3+4)

    vals = GEF.CopySystem() #Create a copy of all initial conditions encoded in GEF
    vals.SetUnits(False)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.kh.SetVal( vals.dphi.value*vals.dI(vals.phi), False)
    yini[3] = np.log(vals.kh.value)

    #currently, all gauge-field expectation values are assumed to be 0 at initialisation
    return yini, vals

def UpdateVals(t, y, vals):
    vals.t.SetVal(t, False)
    vals.N.SetVal(y[0], False)
    vals.a.SetVal(np.exp(y[0]), False)

    vals.phi.SetVal(y[1], False)
    vals.dphi.SetVal(y[2], False)

    vals.kh.SetVal(np.exp(y[3]), False)

    vals.E.SetVal( y[4]*np.exp(4*(y[3]-y[0])), False)
    vals.B.SetVal( y[5]*np.exp(4*(y[3]-y[0])), False)
    vals.G.SetVal( y[6]*np.exp(4*(y[3]-y[0])), False)

    vals.H.SetVal( Friedmann(vals), False)
    vals.xi.SetVal( vals.dI(vals.phi)*(vals.dphi/(2.*vals.H)) )
    vals.ddphi.SetVal(EoMphi(vals))
    return

def TimeStep(t, y, vals, **kwargs):
    atol = kwargs["atol"]
    rtol = kwargs["rtol"]

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = EoMlnkh(vals)
    logfc = y[0] + np.log( 2*abs(vals.xi*dydt[0]) )
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    Fcol = y[4:].shape[0]/3
    F = y[4:].reshape(Fcol,3)
    W = WhittakerApprox(vals.xi.value)
    dFdt = EoMF(vals, F, W, dlnkhdt)
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt






