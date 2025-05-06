import numpy as np
from src.EoMsANDFunctions.ClassicEoMs import *
from src.EoMsANDFunctions.WhittakerFuncs import WhittakerApprox
from src.EoMsANDFunctions.Utility import Heaviside
from src.Solver.Events import Event

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

def UpdateVals(t, y, vals):
    vals.t.SetValue(t)
    vals.N.SetValue(y[0])
    vals.a.SetValue(np.exp(y[0]))

    vals.phi.SetValue(y[1])
    vals.dphi.SetValue(y[2])

    vals.kh.SetValue(np.exp(y[3]))

    vals.E.SetValue( y[4]*np.exp(4*(y[3]-y[0])))
    vals.B.SetValue( y[5]*np.exp(4*(y[3]-y[0])))
    vals.G.SetValue( y[6]*np.exp(4*(y[3]-y[0])))

    vals.H.SetValue( Friedmann(vals))
    vals.xi.SetValue( vals.dI(vals.phi)*(vals.dphi/(2*vals.H)))
    vals.ddphi.SetValue(EoMphi(vals))
    return

def TimeStep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = EoMlnkh(vals)
    logfc = y[0] + np.log( 2*abs(vals.xi)*dydt[0]) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    Fcol = y[4:].shape[0]//3
    F = y[4:].reshape(Fcol,3)
    W = WhittakerApprox(vals.xi.value)
    dFdt = EoMF(vals, F, W, dlnkhdt)
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt

#Event 1:
def EndOfInflationFunc(t, y, vals, atol, rtol):
    dphi = y[2]
    V = vals.V.GetBaseFunc()(vals.MP*y[1])/vals.V.GetConversion()
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


    







