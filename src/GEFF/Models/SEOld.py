import numpy as np

from GEFF.DefaultQuantities import *
from GEFF.GEFSolver import TerminalEvent, ErrorEvent
from GEFF.ModeByMode import ModeSolver
from GEFF.BGTypes import BGVal

from GEFF.Models.EoMsANDFunctions.ClassicEoMs import EoMphi, Friedmann
from GEFF.Models.EoMsANDFunctions.SchwingerEoMs import *
from GEFF.Models.EoMsANDFunctions.WhittakerFuncs import WhittakerApproxSE
from GEFF.Models.EoMsANDFunctions.AuxiliaryFuncs import Heaviside
from GEFF.Models.EoMsANDFunctions.Conductivities import *
from GEFF.Models.EoMsANDFunctions.ModeEoMs import ModeEoMSchwinger, BDDamped



name = "SE-old"

settings = {"pic":"mixed"}

def define_conductivity(setting_dict):
    def collinear_conductivity(frac):
        def conductivity(a, H, E, B, G, H0):
            return ComputeSigmaCollinear(a, H, E, B, G, frac, H0)
        return conductivity

    if setting_dict["pic"]=="mixed":
        return ComputeImprovedSigma
    elif setting_dict["pic"]=="electric":
        return collinear_conductivity(-1.0)
    elif setting_dict["pic"]=="magnetic":
        return collinear_conductivity(1.0)
    else:
        raise Exception("Unknown choice for setting 'picture'")

##### Define Variables #####
############################

#Define all additional variables (besides default variables)
sigmaE=BGVal("sigmaE", 1, 0) #electric damping
sigmaB=BGVal("sigmaB", 1, 0) #magnetic damping 
delta=BGVal("delta", 0, 0) #integrated electric damping
xieff=BGVal("xieff", 0, 0) #effective instability parameter
s=BGVal("s", 0, 0) #electric damping parameter,
rhoChi=BGVal("rhoChi", 4, 0)#Fermion energy density 

# define gauge field by assigning a name, 0th-order quantities and cut-off scale
GF1 = type("GF", (object,), {"name":"GF","0thOrder":{E, B, G}, "UV":kh})

#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "time":{t}, #time coordinate according to which EoMs are expressed
            "dynamical":{N, phi, dphi, kh, delta, rhoChi}, #variables which evolve in time according to an EoM
            "static":{a, H, xi, E, B, G, ddphi, sigmaE, sigmaB, xieff, s}, #variables which are derived from dynamical variables
            "constant":{beta}, #constant quantities in the model
            "function":{V, dV}, #functions of variables such as scalar potentials
            "gaugefields":{GF1} #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
            }

##### Define Input hanlder #####
################################

#State which variables require input for initialisation
input = {
        "initial data":{"phi", "dphi", "rhoChi"},
        "constants":{"beta"},
        "functions":{"V", "dV"}
        }

#Define how initial data is used to infer the initial Hubble rate, Planck mass, and other initial conditions
def parse_input(consts, init, funcs):
    #Compute Hubble rate
    H0 = Friedmann( init["dphi"], funcs["V"](init["phi"]), 0., 0., init["rhoChi"], 1. )
    
    freq = H0 #Characteristic frequency is the initial Hubble rate
    amp = 1. #Charatcterisic amplitude is the Planck mass

    global conductivity
    conductivity = np.vectorize(define_conductivity(settings))

    return freq, amp


##### Define Solver #####
#########################

def initial_conditions(vals, ntr):
    yini = np.zeros((ntr+1)*3+6)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.Initialise("kh")( abs(vals.dphi)*vals.beta )
    yini[3] = np.log(vals.kh.value)

    #initialise delta and rhoChi
    yini[4] = vals.delta.value
    yini[5] = vals.rhoChi.value

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

def update_values(t, y, vals, atol=1e-20, rtol=1e-6):
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

    eps = np.maximum(abs(y[0])*rtol, atol)
    GlobalFerm = Heaviside(np.log(ks/(vals.a*vals.H)), eps)
    vals.sigmaE.SetValue(GlobalFerm*sigmaE)
    vals.sigmaB.SetValue(GlobalFerm*sigmaB)

    vals.s.SetValue(vals.sigmaE/(2*vals.H))
    vals.xi.SetValue( vals.beta*(vals.dphi/(2*vals.H)) )
    vals.xieff.SetValue(vals.xi + vals.sigmaB/(2*vals.H))

    vals.ddphi.SetValue( EoMphi(vals.dphi, vals.dV(vals.phi), vals.beta, vals.G, vals.H, vals.H0) ) 
    return

def compute_timestep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = EoMlnkhSE( vals.kh, vals.dphi, vals.ddphi, vals.beta,
                            0., vals.xieff, vals.s,
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



#Event 1:
def condition_EndOfInflation(t, y, vals):
    ratio = vals.H0/vals.MP
    dphi = y[2]
    V = vals.V(y[1])
    rhoEB = 0.5*(y[6]+y[7])*ratio**2*np.exp(4*(y[3]-y[0]))
    rhoChi = y[5]*ratio**2
    val = np.log(abs((dphi**2 + rhoEB + rhoChi)/V))
    return val

def consequence_EndOfInflation(vals, occurance):
    if occurance:
        return "finish", {}
    else:
        tdiff = np.round(5/vals.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(vals.t + tdiff, 0)

        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
        return "proceed", {"tend":tend}
    
EndOfInflation = TerminalEvent("End of inflation", condition_EndOfInflation, 1, consequence_EndOfInflation)

#Event 2:
def condition_NegativeEnergies(t, y, vals):
    return min(y[6], y[7])
    
NegativeEnergies = ErrorEvent("Negative energies", condition_NegativeEnergies, -1)

events = [EndOfInflation, NegativeEnergies]

##### Define Handling of Gauge Fields #####
###########################################

#define mode-by-mode solver
MbM = ModeSolver(ModeEq=ModeEoMSchwinger, EoMkeys=["a", "xieff", "H", "sigmaE"],
                         BDEq=BDDamped, Initkeys=["a", "delta", "sigmaE"], default_atol=1e-5)