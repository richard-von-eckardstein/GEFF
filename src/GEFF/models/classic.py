"""
Module defining the GEF model "classic" corresponding to pure axion inflation.

For more details on this model, see e.g., [2109.01651](https://arxiv.org/abs/2109.01651).
"""
import numpy as np

from GEFF.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta
from GEFF.solver import TerminalEvent, ErrorEvent, GEFSolver
from GEFF.mode_by_mode import BaseModeSolver

from GEFF.utility.aux_eom import friedmann, gauge_field_ode, dlnkh, klein_gordon
from GEFF.utility.boundary import boundary_approx
from GEFF.utility.auxiliary import heaviside
from GEFF._docs import generate_docs, docs_models


name : str = "classic"
"""The models name."""

settings : dict = {}
"""The model settings."""

# define gauge field by assigning a name, 0th-order quantities and cut-off scale
GF1 = type("GF", (object,), {"name":"GF","0thOrder":{E, B, G}, "UV":kh})

quantities : dict={
            "time":{t}, #time coordinate according to which EoMs are expressed
            "dynamical":{N, phi, dphi, kh}, #variables which evolve in time according to an EoM
            "static":{a, H, xi, E, B, G, ddphi}, #variables which are derived from dynamical variables
            "constant":{beta}, #constant quantities in the model
            "function":{V, dV}, #functions of variables such as scalar potentials
            "gauge":{GF1} #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
            }
r"""The following variables are tracked by the model:

* **time variable**: cosmic time, $t$
* **dynamical variable**:
    * $e$-folds, $N$
    * inflaton amplitude and its velocity, $\varphi$, $\dot{\varphi}$
    * the instability scale $k_{\rm h}$
* **static variables**:
    * scale factor: $a$
    * Hubble rate: $H$
    * instability parameter $\xi$
    * gauge-field expectation values: $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, $-\langle {\bf E} \cdot {\bf B} \rangle$
    * inflaton acceleration, $\ddot{\varphi}$
* **constants**: coupling strength, $\beta$
* **functions**: inflaton potential and its derivative, $V(\varphi)$, $V_{,\varphi}(\varphi)$
* **gauge**: tower of re-scales gauge-bilinears, $\mathcal{F}_{\mathcal X}^{(n)}$, $\mathcal{X} = \mathcal{E}, \mathcal{B}, \mathcal{G}$
"""

#State which variables require input for initialisation
input = {
        "initial data":{"phi", "dphi"},
        "constants":{"beta"},
        "functions":{"V", "dV"}
        }
r"""Define the expected input of the model.

* initial data on the inflaton: $\varphi$, $\dot\varphi$
* coupling strength: $\beta$
* potential shape: $V(\varphi)$, $V_{,\varphi}(\varphi)$
"""

#this functions is called upon initialisation of the GEF class
def define_units(input):
    #compute Hubble rate at t0
    rhoK = input["init"]["dphi"]**2
    rhoV = input["funcs"]["V"](input["init"]["phi"])
    H0 = friedmann( rhoK, rhoV )
    
    freq = H0 #Characteristic frequency is the initial Hubble rate in Planck units
    amp = 1. #Charatcterisic amplitude is the Planck mass (in Planck units)

    return freq, amp

#define vals_to_yini for GEFSolver
def initial_conditions(vals, ntr):
    yini = np.zeros((ntr+1)*3+4)

    #from the 'input' dictionary
    yini[0] = vals.N.value
    yini[1] = vals.phi.value
    yini[2] = vals.dphi.value

    #needs to be computed
    vals.initialise("kh")( abs(vals.dphi)*vals.beta )
    yini[3] = np.log(vals.kh.value)

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

#define update_vals for GEFSolver
def update_values(t, y, vals, atol=1e-20, rtol=1e-6):
    #spacetime variables
    vals.t.set_value(t)
    vals.N.set_value(y[0])
    vals.a.set_value(np.exp(y[0]))

    #parse for convenience
    vals.phi.set_value(y[1])
    vals.dphi.set_value(y[2])
    vals.kh.set_value(np.exp(y[3]))

    #the gauge-field terms in y are not stored, save these values here
    vals.E.set_value( y[4]*np.exp(4*(y[3]-y[0])))
    vals.B.set_value( y[5]*np.exp(4*(y[3]-y[0])))
    vals.G.set_value( y[6]*np.exp(4*(y[3]-y[0])))

    #Hubble rate
    vals.H.set_value( friedmann(0.5*vals.dphi**2, vals.V(vals.phi), 0.5*(vals.E+vals.B)*vals.H0**2) )

    #boundary term parameter
    vals.xi.set_value( vals.beta*(vals.dphi/(2*vals.H)))

    #acceleration for convenience
    vals.ddphi.set_value( klein_gordon(vals.dphi, vals.dV(vals.phi), -vals.G*vals.beta*vals.H0**2)  )
    return

#define timestep for GEFSolver
def compute_timestep(t, y, vals, atol=1e-20, rtol=1e-6):

    dydt = np.zeros(y.shape)

    #odes for N and phi
    dydt[0] = vals.H.value
    dydt[1] = vals.dphi.value
    dydt[2] = vals.ddphi.value

    #achieving dlnkhdt is monotonous requires some care
    dlnkhdt = dlnkh( vals.kh, vals.dphi, vals.ddphi, vals.beta,
                       0., vals.xi, vals.a, vals.H )
    
    logfc = y[0] + np.log( 2*abs(vals.xi)*dydt[0])
    eps = max(abs(y[3])*rtol, atol) 
    dlnkhdt *= heaviside(dlnkhdt, eps)*heaviside(logfc-y[3]+10*eps, eps)
    dydt[3] = dlnkhdt

    #compute boundary terms and then the gauge-field bilinear ODEs
    Fcol = y[4:].shape[0]//3
    F = y[4:].reshape(Fcol,3)
    W = boundary_approx(vals.xi.value)
    dFdt = gauge_field_ode(F, vals.a, vals.kh, 2*vals.H*vals.xi, W, dlnkhdt, L=20)
    #reshape to fit dydt
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt

#Event 1: Track the end of inflation:
def condition_EndOfInflation(t, y, vals):
    dphi = y[2]
    V = vals.V(y[1])
    rhoEB = 0.5*(y[4]+y[5])*(vals.H0/vals.MP)**2*np.exp(4*(y[3]-y[0]))
    val = np.log(abs((dphi**2 + rhoEB)/V))
    return val

def consequence_EndOfInflation(vals, occurrence):
    if occurrence:
        #stop solving once the end of inflation is reached
        return "finish", {}
    else:
        #increase tend given the current Hubble rate
        tdiff = np.round(5/vals.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(vals.t + tdiff, 0)

        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
        return "proceed", {"tend":tend}

EndOfInflation : TerminalEvent = TerminalEvent("End of inflation", condition_EndOfInflation, 1, consequence_EndOfInflation)
"""Defines the 'End of inflation' event."""

#Event 2: ensure energy densities that are positive definite do not become negative
def condition_NegativeEnergies(t, y, vals):
    return min(y[4], y[5])
    
NegativeEnergies : ErrorEvent = ErrorEvent("Negative energies", condition_NegativeEnergies, -1)
"""Defines the 'Negative energy' event."""

events = [EndOfInflation, NegativeEnergies]

#gather all information in the solver
solver = GEFSolver(initial_conditions, update_values, compute_timestep, events, quantities)
"""The solver used by the GEF model."""

#define mode-by-mode solver
MbM = BaseModeSolver
"""The mode solver used by the GEF model."""


#define default docs for the above functions
generate_docs(docs_models.DOCS)





