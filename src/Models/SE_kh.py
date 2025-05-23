import numpy as np
from src.EoMsANDFunctions.ClassicEoMs import EoMphi, Friedmann, EoMlnkh
from src.EoMsANDFunctions.SchwingerEoMs import *
from src.EoMsANDFunctions.WhittakerFuncs import WhittakerApprox
from src.EoMsANDFunctions.AuxiliaryFuncs import Heaviside
from src.EoMsANDFunctions.Conductivities import *
from src.EoMsANDFunctions.ModeEoMs import ModeEoMSchwinger_kS, BDClassic
from src.Solver.Events import Event
from src.Tools.ModeByMode import ModeSolver
from src.BGQuantities.BGTypes import BGVal, BGFunc

name = "SE-kh"

sigmaE=BGVal("sigmaE", 1, 0) #electric damping
sigmaB=BGVal("sigmaB", 1, 0) #magnetic damping 
xieff=BGVal("xieff", 0, 0) #effective instability parameter
rhoChi=BGVal("rhoChi", 4, 0)#Fermion energy density 
kS=BGVal("kS", 1, 0)#Fermion energy density 

modelQuantities = {sigmaE, sigmaB, xieff, rhoChi, kS}   

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
    yini = np.zeros((ntr+1)*3+5)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.kh.SetValue( abs(vals.dphi)*vals.dI(vals.phi))
    yini[3] = np.log(vals.kh.value)

    #initialise delta and rhoChi
    yini[4] = vals.rhoChi.value

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
    vals.kS.SetValue(vals.kh.value)

    vals.rhoChi.SetValue(y[4])

    vals.E.SetValue( y[5]*np.exp(4*(y[3]-y[0])))
    vals.B.SetValue( y[6]*np.exp(4*(y[3]-y[0])))
    vals.G.SetValue( y[7]*np.exp(4*(y[3]-y[0])))

    vals.H.SetValue( Friedmann(vals.dphi, vals.V(vals.phi),
                                 vals.E, vals.B, vals.rhoChi, vals.H0) )

    sigmaE, sigmaB, ks = conductivity(
        vals.a.value, vals.H.value,
          vals.E.value, vals.B.value, vals.G.value
          , vals.H0) # How do I treat model settings?

    eps = np.vectorize(max)(abs(y[0])*rtol, atol)
    GlobalFerm = Heaviside(np.log(ks/(vals.a*vals.H)), eps)
    vals.sigmaE.SetValue(GlobalFerm*sigmaE)
    vals.sigmaB.SetValue(GlobalFerm*sigmaB)

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
    dlnkhdt = EoMlnkh( vals.kh, vals.dphi, vals.ddphi, vals.dI(vals.phi),
                       vals.ddI(vals.phi), vals.xi, vals.a, vals.H )
    r = 2*abs(vals.xi)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    dydt[4] = EoMrhoChi( vals.rhoChi, vals.E, vals.G,
                         vals.sigmaE, vals.sigmaB, vals.H )

    Fcol = y[5:].shape[0]//3
    F = y[5:].reshape(Fcol,3)
    W = WhittakerApprox(vals.xi)
    dFdt = EoMFSE( F, vals.kh, vals.a, 2*vals.H*vals.xieff,
                    vals.sigmaE, 1.0,
                        W, dlnkhdt )
    dydt[5:] = dFdt.reshape(Fcol*3)

    return dydt

ModeByMode = ModeSolver(ModeEq=ModeEoMSchwinger_kS,
                         EoMkeys=["a", "xi", "H", "sigmaE", "sigmaB", "kS"],
                         BDInitEq=BDClassic, Initkeys=[], default_atol=1e-3)

#Event 1:
def EndOfInflationFunc(t, y, vals, atol=1e-20, rtol=1e-6):
    ratio = vals.H0/vals.MP
    dphi = y[2]
    V = vals.V.GetBaseFunc()(vals.MP*y[1])/vals.V.GetConversion()
    rhoEB = 0.5*(y[6]+y[7])*ratio**2*np.exp(4*(y[3]-y[0]))
    rhoChi = y[5]*ratio**2
    val = np.log(abs((dphi**2 + rhoEB + rhoChi)/V))
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