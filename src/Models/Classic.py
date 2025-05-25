import numpy as np
from src.EoMsANDFunctions.ClassicEoMs import *
from src.EoMsANDFunctions.WhittakerFuncs import WhittakerApprox
from src.EoMsANDFunctions.AuxiliaryFuncs import Heaviside
from src.EoMsANDFunctions.ModeEoMs import ModeEoMClassic, BDClassic
from src.Solver.Events import Event
from src.Tools.ModeByMode import ModeSolver

"""
Module defining the model "Classic" used by the GEF

Functions
---------
Friedmann
    compute the Hubble rate
EoMPhi
    the Klein-Gordon equation in presence of gauge-field friction
EoMlnkh
    compute the time derivative of the instability scale kh
EoMF
    compute the time derivatives of the gauge-field bilinear tower
"""

name = "Classic"

modelQuantities = {}
modelFunctions = {}
modelRhos = []

modelSettings = {}

def Initialise(vals, ntr):
    yini = np.zeros((ntr+1)*3+4)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.kh.SetValue( abs(vals.dphi)*vals.dI(vals.phi) )
    yini[3] = np.log(vals.kh.value)

    #currently, all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

def UpdateVals(t, y, vals, atol=1e-20, rtol=1e-6):
    vals.t.SetValue(t)
    vals.N.SetValue(y[0])
    vals.a.SetValue(np.exp(y[0]))

    vals.phi.SetValue(y[1])
    vals.dphi.SetValue(y[2])

    vals.kh.SetValue(np.exp(y[3]))

    vals.E.SetValue( y[4]*np.exp(4*(y[3]-y[0])))
    vals.B.SetValue( y[5]*np.exp(4*(y[3]-y[0])))
    vals.G.SetValue( y[6]*np.exp(4*(y[3]-y[0])))

    vals.H.SetValue( Friedmann(vals.dphi, vals.V(vals.phi),
                                 vals.E, vals.B, 0., vals.H0) )
    
    vals.xi.SetValue( vals.dI(vals.phi)*(vals.dphi/(2*vals.H)))

    vals.ddphi.SetValue( EoMphi(vals.dphi, vals.dV(vals.phi),
                                vals.dI(vals.phi), vals.G, vals.H, vals.H0) )
    return

def TimeStep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    
    dlnkhdt = EoMlnkh( vals.kh, vals.dphi, vals.ddphi, vals.dI(vals.phi),
                       vals.ddI(vals.phi), vals.xi, vals.a, vals.H )
    logfc = y[0] + np.log( 2*abs(vals.xi)*dydt[0])
    eps = max(abs(y[3])*rtol, atol) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    Fcol = y[4:].shape[0]//3
    F = y[4:].reshape(Fcol,3)
    W = WhittakerApprox(vals.xi.value)
    dFdt = EoMF(F, vals.a, vals.kh, 2*vals.H*vals.xi, W, dlnkhdt)
    
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt

ModeByMode = ModeSolver(ModeEq=ModeEoMClassic, EoMkeys=["a", "xi", "H"],
                         BDEq=BDClassic, Initkeys=[], default_atol=1e-3)

#Event 1:
def EndOfInflationFunc(t, y, vals, atol, rtol):
    dphi = y[2]
    V = vals.V(y[1])
    rhoEB = 0.5*(y[4]+y[5])*(vals.H0/vals.MP)**2*np.exp(4*(y[3]-y[0]))
    val = np.log(abs((dphi**2 + rhoEB)/V))
    return val

def EndOfInflationConsequence(vals, occurance):
    if occurance:
        return {"primary":"finish"}
    else:
        tdiff = np.round(5/vals.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(vals.t + tdiff, 0)

        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
        return {"primary":"proceed", "secondary":{"tend":tend}}
    
EndOfInflation = Event("End of inflation", EndOfInflationFunc, True, 1, EndOfInflationConsequence)

events = [EndOfInflation]


    







