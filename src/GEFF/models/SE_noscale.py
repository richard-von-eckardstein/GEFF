r"""
Module defining the GEF model "SE no-scale" corresponding to fermionic axion inflation without a heuristic scale dependence.

For more details on this model, see e.g., [2109.01651](https://arxiv.org/abs/2109.01651).
"""
import numpy as np

from GEFF.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta, BGVal
from GEFF.solver import TerminalEvent, ErrorEvent
from GEFF.mode_by_mode import ModeSolver

from GEFF.utility.aux_eom import (klein_gordon, friedmann, dlnkh_schwinger,
                                      ddelta, drhoChi, gauge_field_ode_schwinger,
                                        conductivities_collinear, conductivities_mixed)
from GEFF.utility.boundary import boundary_approx_schwinger
from GEFF.utility.auxiliary import heaviside
from GEFF.utility.aux_mode import mode_equation_SE_scale, damped_bd

name = "SE no-scale"

settings = {"pic":"mixed"}

def define_conductivity(setting_dict):
    def collinear_conductivity(frac):
        def conductivity(a, H, E, B, G, H0):
            return conductivities_collinear(a, H, E, B, G, frac, H0)
        return conductivity

    if setting_dict["pic"]=="mixed":
        return conductivities_mixed
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
            "gauge":{GF1} #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
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
def define_units(consts, init, funcs):
    #Compute Hubble rate
    H0 = friedmann(  0.5*init["dphi"]**2, funcs["V"](init["phi"]), init["rhoChi"] )
    
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

    vals.initialise("kh")( abs(vals.dphi)*vals.beta )
    yini[3] = np.log(vals.kh.value)

    #initialise delta and rhoChi
    yini[4] = vals.delta.value
    yini[5] = vals.rhoChi.value

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

def update_values(t, y, vals, atol=1e-20, rtol=1e-6):
    vals.t.set_value(t)
    vals.N.set_value(y[0])
    vals.a.set_value(np.exp(y[0]))

    vals.phi.set_value(y[1])
    vals.dphi.set_value(y[2])

    vals.kh.set_value(np.exp(y[3]))

    vals.delta.set_value(y[4])
    vals.rhoChi.set_value(y[5])

    vals.E.set_value( y[6]*np.exp(4*(y[3]-y[0])))
    vals.B.set_value( y[7]*np.exp(4*(y[3]-y[0])))
    vals.G.set_value( y[8]*np.exp(4*(y[3]-y[0])))

    vals.H.set_value( friedmann(0.5*vals.dphi**2, vals.V(vals.phi), 
                                0.5*(vals.E+vals.B)*vals.H0**2, vals.rhoChi*vals.H0**2) )

    sigmaE, sigmaB, ks = conductivity(
        vals.a.value, vals.H.value,
          vals.E.value, vals.B.value, vals.G.value
          , vals.H0)

    eps = np.maximum(abs(y[0])*rtol, atol)
    GlobalFerm = heaviside(np.log(ks/(vals.a*vals.H)), eps)
    vals.sigmaE.set_value(GlobalFerm*sigmaE)
    vals.sigmaB.set_value(GlobalFerm*sigmaB)

    vals.s.set_value(vals.sigmaE/(2*vals.H))
    vals.xi.set_value( vals.beta*(vals.dphi/(2*vals.H)) )
    vals.xieff.set_value(vals.xi + vals.sigmaB/(2*vals.H))

    vals.ddphi.set_value( klein_gordon(vals.dphi, vals.dV(vals.phi), -vals.G*vals.beta*vals.H0**2) )
    return

def compute_timestep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = dlnkh_schwinger( vals.kh, vals.dphi, vals.ddphi, vals.beta,
                                        0., vals.xieff, vals.s, vals.a, vals.H )

    xieff = vals.xieff.value
    s = vals.s.value
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= heaviside(dlnkhdt, eps)*heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    dydt[4] = ddelta( vals.delta, vals.sigmaE )
    dydt[5] = drhoChi( vals.rhoChi, vals.E, vals.G, vals.sigmaE, vals.sigmaB, vals.H )

    Fcol = y[6:].shape[0]//3
    F = y[6:].reshape(Fcol,3)
    W = boundary_approx_schwinger(vals.xieff.value, vals.s.value)
    dFdt = gauge_field_ode_schwinger( F, vals.a, vals.kh, 2*vals.H*vals.xieff,
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
MbM = ModeSolver(mode_equation_SE_scale, ["a", "xieff", "H", "sigmaE"],
                         damped_bd, ["a", "sigmaE", "delta"], default_atol=1e-3)