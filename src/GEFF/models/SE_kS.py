r"""
Module defining the GEF model "SE no-scale" corresponding to fermionic axion inflation without a heuristic scale dependence.

For more details on this model, see e.g., [2109.01651](https://arxiv.org/abs/2109.01651).
"""
import numpy as np

from GEFF.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta, BGVal, BGFunc
from GEFF.solver import TerminalEvent, ErrorEvent, GEFSolver
from GEFF.mbm import ModeSolver

from GEFF.utility.aux_eom import (klein_gordon, friedmann, dlnkh_schwinger,
                                      ddelta, drhoChi, gauge_field_ode_schwinger,
                                        conductivities_collinear, conductivities_mixed)
from GEFF.utility.boundary import boundary_approx_schwinger
from GEFF.utility.auxiliary import heaviside
from GEFF.utility.aux_mode import mode_equation_SE_no_scale, damped_bd
from GEFF._docs import generate_docs, docs_models

name = "SE-kS"
"""The models name."""

settings = {"pic":"mixed"}
"""The model settings.

Possible settings are "mixed", "electric", "magnetic".

Determines if conductivities are computed assuming collinear E&M fields 
("electric", "magnetic") or not ("mixed").
"""

# parse settings
def define_conductivity():
    if settings["pic"]=="mixed":
        conductivity = conductivities_mixed
    elif settings["pic"]=="electric":
        def conductivity(a, H, E, B, G, H0):
            return conductivities_collinear(a, H, E, B, G, -1, H0)
    elif settings["pic"]=="magnetic":
        def conductivity(a, H, E, B, G, H0):
            return conductivities_collinear(a, H, E, B, G, 1, H0)
    else:
        raise KeyError(f"{settings['pic']} is an unknown choice for the setting'pic'")
    return np.vectorize(conductivity)

def interpret_settings():
    global conductivity
    conductivity = define_conductivity()
    return

#Define all additional variables
sigmaE=BGVal("sigmaE", 1, 0) #electric damping
sigmaB=BGVal("sigmaB", 1, 0) #magnetic damping 
deltaF=BGFunc("deltaF", [t], 0, 0) #integrated electric damping, func not par
xieff=BGVal("xieff", 0, 0) #effective instability parameter
s=BGVal("s", 0, 0) #electric damping parameter,
rhoChi=BGVal("rhoChi", 4, 0)#Fermion energy density 
kS = BGVal("kS", 1, 0) # pair creation scale
EBar = BGVal("EBar", 4, 0)
BBar = BGVal("BBar", 4, 0)
GBar = BGVal("GBar", 4, 0)

# define gauge field by assigning a name, 0th-order quantities and cut-off scale
GF1 = type("GF", (object,), {"name":"GF","0thOrder":{E, B, G}, "UV":kh})
GFBar = type("GF", (object,), {"name":"GF","0thOrder":{EBar, BBar, GBar}, "UV":kS})

#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "time":{t},
            "dynamical":{N, phi, dphi, kh, rhoChi},
            "static":{a, H, xi, kS, E, B, G, ddphi, sigmaE, sigmaB, xieff, s},
            "constant":{beta},
            "function":{V, dV, deltaF},
            "gauge":{GF1, GFBar}
            }
r"""The following variables are tracked by the model:

* **time variable**: cosmic time, $t$
* **dynamical variable**:
    * $e$-folds, $N$
    * inflaton amplitude and its velocity, $\varphi$, $\dot{\varphi}$
    * the instability scale $k_{\rm h}$
    * fermion energy density, $\rho_{\chi}$
    * cumulative electric damping, $\Delta$
* **static variables**:
    * scale factor: $a$
    * Hubble rate: $H$
    * instability parameter $\xi$
    * gauge-field expectation values: $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, $-\langle {\bf E} \cdot {\bf B} \rangle$
    * inflaton acceleration, $\ddot{\varphi}$
    * electric and magnetic conductivities $\sigma_{\rm E/B}$
    * effective instability parameter $\xi_{\rm eff}$
    * effective electric damping, $s = \sigma_{\rm E}/(2H)$
* **constants**: coupling strength, $\beta$
* **functions**: inflaton potential and its derivative, $V(\varphi)$, $V_{,\varphi}(\varphi)$
* **gauge**: tower of re-scales gauge-bilinears, $\mathcal{F}_{\mathcal X}^{(n)}$, $\mathcal{X} = \mathcal{E}, \mathcal{B}, \mathcal{G}$
"""

#State which variables require input for initialisation
input = {
        "initial data":{"phi", "dphi", "rhoChi"},
        "constants":{"beta"},
        "functions":{"V", "dV"}
        }
r"""Define the expected input of the model.

* initial data on the inflaton: $\varphi$, $\dot\varphi$
* initial data on the fermionic energy density: $\rho_{\chi}$
* coupling strength: $\beta$
* potential shape: $V(\varphi)$, $V_{,\varphi}(\varphi)$
"""

#this functions is called upon initialisation of the GEF class
def define_units(consts, init, funcs):
    #compute Hubble rate at t0
    rhoK = 0.5*init["dphi"]**2
    rhoV = funcs["V"](init["phi"])
    rhochi = init["rhoChi"]
    H0 = friedmann( rhoK, rhoV, rhochi )
    
    freq = H0 #Characteristic frequency is the initial Hubble rate
    amp = 1. #Charatcterisic amplitude is the Planck mass

    return freq, amp

#the new function for sys_to_yini in GEFSolver
def initial_conditions(sys, ntr):
    yini = np.zeros((ntr+1)*6+5)

    #from the 'input' dictionary
    yini[0] = sys.N.value
    yini[1] = sys.phi.value
    yini[2] = sys.dphi.value

    #needs to be computed
    sys.initialise("kh")( abs(sys.dphi)*sys.beta )
    yini[3] = np.log(sys.kh.value)

    #initialise delta and rhoChi
    sys.initialise("deltaF")( lambda x: 1.0 )
    yini[4] = sys.rhoChi.value

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

#define update_sys for GEFSolver
def update_values(t, y, sys, atol=1e-20, rtol=1e-6):
    #spacetime variables
    sys.t.set_value(t)
    sys.N.set_value(y[0])
    sys.a.set_value(np.exp(y[0]))

    #parse for convenience
    sys.phi.set_value(y[1])
    sys.dphi.set_value(y[2])
    sys.kh.set_value(np.exp(y[3]))
    sys.rhoChi.set_value(y[4])

    #the gauge-field terms in y are not stored, save these values here
    sys.E.set_value( y[6]*np.exp(4*(y[3]-y[0])))
    sys.B.set_value( y[7]*np.exp(4*(y[3]-y[0])))
    sys.G.set_value( y[8]*np.exp(4*(y[3]-y[0])))

    #Hubble rate
    sys.H.set_value( friedmann(0.5*sys.dphi**2, sys.V(sys.phi), 
                                0.5*(sys.E+sys.B)*sys.H0**2, sys.rhoChi*sys.H0**2) )

    #conductivities
    sigmaE, sigmaB, kF = conductivity(sys.a.value, sys.H.value, sys.E.value,
                                       sys.B.value, sys.G.value, sys.H0)
    eps = np.maximum(abs(y[0])*rtol, atol)
    GlobalFerm = heaviside(np.log(kF/(sys.a*sys.H)), eps)
    sys.sigmaE.set_value(GlobalFerm*sigmaE)
    sys.sigmaB.set_value(GlobalFerm*sigmaB)
    sys.kS.set_value(kF)

    sys.Ebar.set_value( np.sign(y[9])*min(abs(y[9]*(kF/sys.a)**4), abs(sys.E.value*(kF/sys.kh.value)**4)))
    sys.Bbar.set_value( np.sign(y[10])*min(abs(y[10]*(kF/sys.a)**4), abs(sys.B.value*(kF/sys.kh.value)**4)))
    sys.Gbar.set_value( np.sign(y[11])*min(abs(y[11]*(kF/sys.a)**4), abs(sys.G.value*(kF/sys.kh.value)**4)))


    #boundary term parameters
    sys.s.set_value(sys.sigmaE/(2*sys.H))
    sys.xi.set_value( sys.beta*(sys.dphi/(2*sys.H)) )
    sys.xieff.set_value(sys.xi + sys.sigmaB/(2*sys.H))

    #acceleration for convenience
    sys.ddphi.set_value( klein_gordon(sys.dphi, sys.dV(sys.phi), sys.H, -sys.G*sys.beta*sys.H0**2) )
    return

def compute_timestep(t, y, sys, atol=1e-20, rtol=1e-6):
    dydt = np.zeros(y.shape)

    #odes for N and phi
    dydt[0] = sys.H.value
    dydt[1] = sys.dphi.value
    dydt[2] = sys.ddphi.value

    #achieving dlnkhdt is monotonous requires some care
    eps = max(abs(y[3])*rtol, atol)
    dlnkhdt = dlnkh_schwinger( sys.kh, sys.dphi, sys.ddphi, sys.beta,
                                        0., sys.xieff, sys.s, sys.a, sys.H )
    xieff = sys.xieff.value
    s = sys.s.value
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= heaviside(dlnkhdt, eps)*heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    #the other derivatives are straight forwards
    dydt[4] = drhoChi( sys.rhoChi, sys.EBar, sys.GBar, sys.sigmaE, sys.sigmaB, sys.H )

    #compute boundary terms and then the gauge-field bilinear ODEs
    Fcol = y[5:].shape[0]//3
    F = y[5:].reshape(Fcol,3)
    W = boundary_approx_schwinger(sys.xieff.value, sys.s.value)
    #TODO
    dFdt = gauge_field_ode_schwinger( F, sys.a, sys.kh, 2*sys.H*sys.xieff,
                                            sys.sigmaE, sys.delta,
                                                W, dlnkhdt )
    #reshape to fit dydt
    dydt[6:] = dFdt.reshape(Fcol*3)

    return dydt

def compute_timestep_Cross(t, y, sys, atol=1e-20, rtoL=1e-6):
    return



#Event 1: Track the end of inflation:
def condition_EndOfInflation(t, y, sys):
    ratio = sys.H0/sys.MP
    dphi = y[2]
    V = sys.V(y[1])
    rhoEB = 0.5*(y[6]+y[7])*ratio**2*np.exp(4*(y[3]-y[0]))
    rhoChi = y[5]*ratio**2
    val = np.log(abs((dphi**2 + rhoEB + rhoChi)/V))
    return val

def consequence_EndOfInflation(sys, occurance):
    if occurance:
        #stop solving once the end of inflation is reached
        return "finish", {}
    else:
        #increase tend given the current Hubble rate
        tdiff = np.round(5/sys.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(sys.t + tdiff, 0)

        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
        return "proceed", {"tend":tend}
    
EndOfInflation = TerminalEvent("End of inflation", condition_EndOfInflation, 1, consequence_EndOfInflation)
"""Defines the 'End of inflation' event."""

#Event 2: ensure energy densities that are positive definite do not become negative
def condition_NegativeEnergies(t, y, sys):
    return min(y[6], y[7])
    
NegativeEnergies = ErrorEvent("Negative energies", condition_NegativeEnergies, -1)
"""Defines the 'Negative energy' event."""

events = [EndOfInflation, NegativeEnergies]


#gather all information in the solver
solver = GEFSolver(initial_conditions, update_values, compute_timestep, quantities, events)
"""The solver used by the GEF model."""

#define mode-by-mode solver
MbM = ModeSolver(mode_equation_SE_no_scale, ["a", "xieff", "H", "sigmaE"],
                         damped_bd, ["a", "sigmaE", "delta"], default_atol=1e-5)
"""The mode solver used by the GEF model."""


#define default docs for the above functions
generate_docs(docs_models.DOCS)