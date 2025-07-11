import numpy as np
from src.EoMsANDFunctions.ClassicEoMs import EoMphi, Friedmann
from src.EoMsANDFunctions.SchwingerEoMs import *
from src.EoMsANDFunctions.WhittakerFuncs import WhittakerApproxSE
from src.EoMsANDFunctions.AuxiliaryFuncs import Heaviside
from src.EoMsANDFunctions.Conductivities import *
from src.EoMsANDFunctions.ModeEoMs import ModeEoMSchwinger, BDDamped
from src.Solver.Events import Event
from src.Tools.ModeByMode import ModeSolver
from src.BGQuantities.BGTypes import BGVal, BGFunc

name = "SE-Old"

sigmaE=BGVal("sigmaE", 1, 0) #electric damping
sigmaB=BGVal("sigmaB", 1, 0) #magnetic damping 
delta=BGVal("delta", 0, 0) #integrated electric damping
xieff=BGVal("xieff", 0, 0) #effective instability parameter
s=BGVal("s", 0, 0) #electric damping parameter,
rhoChi=BGVal("rhoChi", 4, 0)#Fermion energy density 

modelQuantities = {sigmaE, sigmaB, delta, xieff, s, rhoChi}   

modelFunctions = {}

modelRhos = ["rhoChi"]

modelSettings = {"pic": "mixed"}

def DefineConductivity(settings):
    def CollinearConductivity(frac):
        def conductivity(a, H, E, B, G, H0):
            return ComputeSigmaCollinear(a, H, E, B, G, frac, H0)
        return conductivity

    if settings["pic"]=="mixed":
        return ComputeImprovedSigma
    elif settings["pic"]=="electric":
        return CollinearConductivity(-1.0)
    elif settings["pic"]=="magnetic":
        return CollinearConductivity(1.0)
    else:
        print("Hi there")
        raise Exception("Unknown choice for setting 'picture'")

def Initialise(vals, ntr):
    yini = np.zeros((ntr+1)*3+6)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.kh.SetValue( abs(vals.dphi)*vals.dI(vals.phi))
    yini[3] = np.log(vals.kh.value)

    #initialise delta and rhoChi
    yini[4] = vals.delta.value
    yini[5] = vals.rhoChi.value

    global conductivity
    conductivity = np.vectorize(DefineConductivity(modelSettings))

    #currently, all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

def UpdateVals(t, y, vals, atol=1e-20, rtol=1e-6):
    vals.t.SetValue(t)
    vals.N.SetValue(y[0])
    vals.a.SetValue(np.exp(y[0]))

    vals.phi.SetValue(y[1])
    vals.dphi.SetValue(y[2])

    vals.kh.SetValue(np.exp(y[3]))

    vals.delta.SetValue(y[4])
    vals.rhoChi.SetValue(y[5])

    vals.E.SetValue( y[6]*np.exp(4*(y[3]-y[0])))
    vals.B.SetValue( y[7]*np.exp(4*(y[3]-y[0])))
    vals.G.SetValue( y[8]*np.exp(4*(y[3]-y[0])))

    vals.H.SetValue( Friedmann(vals.dphi, vals.V(vals.phi),
                                 vals.E, vals.B, vals.rhoChi, vals.H0) )

    sigmaE, sigmaB, ks = conductivity(
        vals.a.value, vals.H.value,
          vals.E.value, vals.B.value, vals.G.value
          , vals.H0)

    eps = np.vectorize(max)(abs(y[0])*rtol, atol)
    GlobalFerm = Heaviside(np.log(ks/(vals.a*vals.H)), eps)
    vals.sigmaE.SetValue(GlobalFerm*sigmaE)
    vals.sigmaB.SetValue(GlobalFerm*sigmaB)

    vals.s.SetValue(vals.sigmaE/(2*vals.H))
    vals.xi.SetValue( vals.dI(vals.phi)*(vals.dphi/(2*vals.H)))
    vals.xieff.SetValue(vals.xi + vals.sigmaB/(2*vals.H))

    vals.ddphi.SetValue( EoMphi(vals.dphi, vals.dV(vals.phi),
                                vals.dI(vals.phi), vals.G, vals.H, vals.H0) )
    return

def TimeStep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = EoMlnkhSE( vals.kh, vals.dphi, vals.ddphi, vals.dI(vals.phi),
                            vals.ddI(vals.phi), vals.xieff, vals.s,
                              vals.a, vals.H )
    xieff = vals.xieff.value
    s = vals.s.value
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    dydt[4] = EoMDelta( vals.delta, vals.sigmaE )
    dydt[5] = EoMrhoChi( vals.rhoChi, vals.E, vals.G,
                         vals.sigmaE, vals.sigmaB, vals.H )

    Fcol = y[6:].shape[0]//3
    F = y[6:].reshape(Fcol,3)
    W = WhittakerApproxSE(vals.xieff.value, vals.s.value)
    dFdt = EoMFSE( F, vals.a,
                  vals.kh, 2*vals.H*vals.xieff,
                    vals.sigmaE, vals.delta,
                        W, dlnkhdt )
    dydt[6:] = dFdt.reshape(Fcol*3)

    return dydt

ModeByMode = ModeSolver(ModeEq=ModeEoMSchwinger, EoMkeys=["a", "xieff", "H", "sigmaE"],
                         BDEq=BDDamped, Initkeys=["a", "delta", "sigmaE"], default_atol=1e-5)

#Event 1:
def EndOfInflation_Condition(t, y, vals, atol=1e-20, rtol=1e-6):
    ratio = vals.H0/vals.MP
    dphi = y[2]
    V = vals.V.GetBaseFunc()(vals.MP*y[1])/vals.V.GetConversion()
    rhoEB = 0.5*(y[6]+y[7])*ratio**2*np.exp(4*(y[3]-y[0]))
    rhoChi = y[5]*ratio**2
    val = np.log(abs((dphi**2 + rhoEB + rhoChi)/V))
    return val

def EndOfInflation_Consequence(vals, occurance):
    if occurance:
        return {"primary":"finish"}
    else:
        tdiff = np.round(5/vals.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(vals.t + tdiff, 0)

        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
        return {"primary":"proceed", "secondary":{"tend":tend}}
    
EndOfInflation = Event("End of inflation",
                        EndOfInflation_Condition, True, 1, EndOfInflation_Consequence)

#Event 2:
def NegativeEnergies_Condition(t, y, vals, atol=1e-20, rtol=1e-6):
    return min(y[6], y[7])

def NegativeEnergies_Consequence(vals, occurance):
    if occurance:
        return {"primary":"finish"}
    else:
        return {"primary":"proceed"}
    
NegativeEnergies = Event("Negative energies",
                          NegativeEnergies_Condition, True, -1, NegativeEnergies_Consequence)

events = [EndOfInflation, NegativeEnergies]