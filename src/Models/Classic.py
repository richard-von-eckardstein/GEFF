import numpy as np
from src.BGQuantities.Variables import *

from src.EoMsANDFunctions.ClassicEoMs import *
from src.EoMsANDFunctions.WhittakerFuncs import WhittakerApprox
from src.EoMsANDFunctions.AuxiliaryFuncs import Heaviside
from src.EoMsANDFunctions.ModeEoMs import ModeEoMClassic, BDClassic
from src.Solver.Events import Event
from src.Tools.ModeByMode import ModeSolver

"""
Module defining the model "Classic" used by the GEF
"""

name = "Classic"

settings = {}


##### Define Variables #####
############################

#Define all additional variables (besides default variables)

#dynamical variables
phi = BGVal("phi", 0, 1) #
dphi = BGVal("dphi", 1, 1) #
lnkh = BGVal("lnkh", 0, 0) #log of instability scale

#static variables (which are given are derived from dynamical variables)
kh = BGVal("kh", 1, 0) #log of instability scale
xi = BGVal("xi", 0, 0) #instability parameter
E = BGVal("E", 4, 0) #E^2 expectation value
B = BGVal("B", 4, 0) #B^2 expectation value
G = BGVal("G", 4, 0) #G^2 expecation value

#Define all constants
beta = BGVal("beta", 0, -1)

#Define all functions (potentials, couplings etc.)
V = BGFunc("V", [phi], 2, 2)
dV = BGFunc("dV", [phi], 2, 1)


#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "dynamical":{phi, dphi, lnkh},
            "static":{xi, kh, E, B, G},
            "constant":{beta},
            "function":{V, dV}
            }

##### Define Handling of Gauge Fields #####
###########################################

#define mode-by-mode solver
MbM = ModeSolver(ModeEq=ModeEoMClassic, EoMVals=["a", "xi", "H"],
                         BDEq=BDClassic, Initkeys=[], default_atol=1e-3)

# define gauge field by assigning a name, 0th-order quantities, cut-off scale, and mode-by-mode solver
GF1 = {"GF":{"0thOrder":{E, B, G}, "UV":kh}}
gaugefields = {GF1}

##### Define Input hanlder #####
################################

#State which variables require input for initialisation
input = {
        "dynamic":{phi, dphi},
          "constant":{beta},
        "function":{V, dV}
        }

#Define how initial data is used to infer the initial Hubble rate, Planck mass, and other initial conditions
def ParseInput(consts, init, funcs):
    #Compute Hubble rate
    H0 = np.sqrt( Friedmann( init["dphi"], funcs["V"](init["phi"]), 0., 0., 0., 0. ) )
    
    freq = H0 #Characteristic frequency is the initial Hubble rate
    Amp = 1. #Charatcterisic amplitude is the Planck mass

    #Initialise tose dynamical variables not infered from the input
    derivedInput = {"lnkh": np.log(consts["beta"]*init["dphi"]/(2*H0)), "GF":0}

    init = init.update(derivedInput)

    return freq, Amp


##### Define Solver #####
#########################

#Inform the solver how to compute static variables based on dynamical variables
def ComputeStaticVariables(sol):

    #Compute kh from logkh
    sol.kh.SetValue(np.exp(sol.lnkh))

    #Compute E, B, G from GF
    sol.E.SetValue( sol.GF[0,0]*np.exp( 4*(sol.lnkh -sol.a) ) )
    sol.B.SetValue( sol.GF[0,1]*np.exp( 4*(sol.lnkh -sol.a) ) )
    sol.G.SetValue( sol.GF[0,2]*np.exp( 4*(sol.lnkh -sol.a) ) )

    #Compute Hubble rate
    sol.H.SetValue( Friedmann(sol.dphi, sol.V(sol.phi), sol.E, sol.B, 0., sol.H0) )
    
    #Compute instability parameter
    sol.xi.SetValue( sol.dI(sol.phi)*(sol.dphi/(2*sol.H)))

    return

#Tell the Solver how to compute time derivatives of dynamical variables
def EoM(sol):
    rtol = sol.rtol
    atol = sol.atol

    #Evolve phi field
    sol.Evolve("phi")(sol.dphi)

    ddphi = EoMphi(sol.dphi, sol.dV(sol.phi), sol.dI(sol.phi), sol.G, sol.H, sol.H0)
    sol.Evolve("dphi")(ddphi)

    #Compute dlnkh/dt
    dlnkhdt = EoMlnkh( sol.kh, sol.dphi, ddphi, sol.dI(sol.phi),
                       sol.ddI(sol.phi), sol.xi, sol.a, sol.H )
    logfc = np.log( 2*abs(sol.xi)*sol.H*sol.a )
    eps = max(abs(sol.lnkh)*rtol, atol) 
    dlnkhdt *= Heaviside(dlnkhdt, eps)*Heaviside(logfc-sol.lnkh[3]+10*eps, eps)
    
    #Evolve dlnkh
    sol.Evolve("lnkh")( dlnkhdt )

    #Compute gauge-field tower derivative
    W = WhittakerApprox(sol.xi.value)
    dFdt = EoMF(sol.GF, sol.a, sol.kh, 2*sol.H*sol.xi, W, dlnkhdt, L=20)

    #Evolve gauge-field tower
    sol.Evolve("GF")(dFdt)

    return



#Event 1:
def EndOfInflation_Condition(t, y, vals, atol, rtol):
    dphi = y[2]
    V = vals.V(y[1])
    rhoEB = 0.5*(y[4]+y[5])*(vals.H0/vals.MP)**2*np.exp(4*(y[3]-y[0]))
    val = np.log(abs((dphi**2 + rhoEB)/V))
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
def NegativeEnergies_Condition(t, y, vals, atol, rtol):
    return min(y[4], y[5])

def NegativeEnergies_Consequence(vals, occurance):
    if occurance:
        return {"primary":"finish"}
    else:
        return {"primary":"proceed"}
    
NegativeEnergies = Event("Negative energies",
                          NegativeEnergies_Condition, True, -1, NegativeEnergies_Consequence)

events = [EndOfInflation, NegativeEnergies]


    







