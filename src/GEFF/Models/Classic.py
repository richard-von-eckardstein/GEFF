import numpy as np

from GEFF.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta
from GEFF.GEFSolver import TerminalEvent, ErrorEvent
from GEFF.mode_by_mode import BaseModeSolver

from GEFF.Models.EoMsANDFunctions.ClassicEoMs import *
from GEFF.Models.EoMsANDFunctions.WhittakerFuncs import WhittakerApprox
from GEFF.Models.EoMsANDFunctions.AuxiliaryFuncs import Heaviside


"""
Module defining the model "classic" used by GEF to define a class including a custom GEFSolver and ModeSolver
"""

name = "classic"

settings = {}


##### Define Variables #####
############################

#Define all additional variables (besides default variables)

# define gauge field by assigning a name, 0th-order quantities and cut-off scale
GF1 = type("GF", (object,), {"name":"GF","0thOrder":{E, B, G}, "UV":kh})


#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "time":{t}, #time coordinate according to which EoMs are expressed
            "dynamical":{N, phi, dphi, kh}, #variables which evolve in time according to an EoM
            "static":{a, H, xi, E, B, G, ddphi}, #variables which are derived from dynamical variables
            "constant":{beta}, #constant quantities in the model
            "function":{V, dV}, #functions of variables such as scalar potentials
            "gaugefields":{GF1} #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
            }


##### Define Input hanlder #####
################################

#State which variables require input for initialisation
input = {
        "initial data":{"phi", "dphi"},
        "constants":{"beta"},
        "functions":{"V", "dV"}
        }

#Define how initial data is used to infer the initial Hubble rate, Planck mass, and other initial conditions
def parse_input(consts, init, funcs):
    #Compute Hubble rate
    H0 = Friedmann( init["dphi"], funcs["V"](init["phi"]), 0., 0., 0., 1. )
    
    freq = H0 #Characteristic frequency is the initial Hubble rate
    amp = 1. #Charatcterisic amplitude is the Planck mass

    return freq, amp


##### Define Solver #####
#########################

#Inform the solver how to compute static variables based on dynamical variables
def initial_conditions(vals, ntr):
    yini = np.zeros((ntr+1)*3+4)

    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    vals.initialise("kh")( abs(vals.dphi)*vals.beta )
    yini[3] = np.log(vals.kh.value)

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

def update_values(t, y, vals, atol=1e-20, rtol=1e-6):
    vals.t.set_value(t)
    vals.N.set_value(y[0])
    vals.a.set_value(np.exp(y[0]))

    vals.phi.set_value(y[1])
    vals.dphi.set_value(y[2])

    vals.kh.set_value(np.exp(y[3]))

    vals.E.set_value( y[4]*np.exp(4*(y[3]-y[0])))
    vals.B.set_value( y[5]*np.exp(4*(y[3]-y[0])))
    vals.G.set_value( y[6]*np.exp(4*(y[3]-y[0])))

    vals.H.set_value( Friedmann(vals.dphi, vals.V(vals.phi),
                                 vals.E, vals.B, 0., vals.H0) )
    
    vals.xi.set_value( vals.beta*(vals.dphi/(2*vals.H)))

    vals.ddphi.set_value( EoMphi(vals.dphi, vals.dV(vals.phi), vals.beta, vals.G, vals.H, vals.H0)  )

    return

def compute_timestep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    
    dlnkhdt = EoMlnkh( vals.kh, vals.dphi, vals.ddphi, vals.beta,
                       0., vals.xi, vals.a, vals.H )
    
    logfc = y[0] + np.log( 2*abs(vals.xi)*dydt[0])
    eps = max(abs(y[3])*rtol, atol) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    Fcol = y[4:].shape[0]//3
    F = y[4:].reshape(Fcol,3)
    W = WhittakerApprox(vals.xi.value)
    dFdt = EoMF(F, vals.a, vals.kh, 2*vals.H*vals.xi, W, dlnkhdt, L=20)
    
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt



#Event 1:
def condition_EndOfInflation(t, y, vals):
    dphi = y[2]
    V = vals.V(y[1])
    rhoEB = 0.5*(y[4]+y[5])*(vals.H0/vals.MP)**2*np.exp(4*(y[3]-y[0]))
    val = np.log(abs((dphi**2 + rhoEB)/V))
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
    
EndOfInflation= TerminalEvent("End of inflation", condition_EndOfInflation, 1, consequence_EndOfInflation)

#Event 2:
def condition_NegativeEnergies(t, y, vals):
    return min(y[4], y[5])
    
NegativeEnergies = ErrorEvent("Negative energies", condition_NegativeEnergies, -1)

events = [EndOfInflation, NegativeEnergies]

##### Define Handling of Gauge Fields #####
###########################################

#define mode-by-mode solver
MbM = BaseModeSolver







