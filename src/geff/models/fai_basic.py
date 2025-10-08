r"""
Module defining the GEF model "SE no-scale" corresponding to fermionic axion inflation without a heuristic scale dependence.

For more details on this model, see e.g., [2109.01651](https://arxiv.org/abs/2109.01651).

---

The model knows the following variables:
* time variable: `t` - *cosmic time*, $t$ 
* dynamical variables:
    * `N` - *$e$-folds*,  $N$
    * `phi`, `dphi` - *inflaton amplitude, $\varphi$, and velocity, $\dot{\varphi}$* 
    * `kh` -  *the instability scale, $k_{\rm h}$*
    * `delta` - *cumulative electric damping, $\Delta$*
    * `rhoChi` - *fermion energy density, $\rho_{\chi}$*
* static variables:
    * `a` - *scale factor, $a$* 
    * `H` - *Hubble rate, $H$* 
    * `ddphi` - *inflaton acceleration, $\ddot{\varphi}$*
    * `E`, `B`, `G` - *gauge-field expectation values, $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, -$\langle {\bf E} \cdot {\bf B} \rangle$*
    * `xi` - *instability parameter, $\xi$* 
    * `sigmaE`, `sigmaB` - *electric and magnetic conductivities, $\sigma_{\rm E}$, $\sigma_{\rm B}$*
    * `xieff` - *effective instability parameter, $\xi_{\rm eff}$*
    * `s` - *electric damping parameter, $s = \sigma_{\rm E}/(2H)$*
* constants: 
    * `beta` - *coupling strength, $\beta$*
* functions: 
    * `V`,`dV` - *inflaton potential, $V(\varphi)$, and its derivative, $V_{,\varphi}(\varphi)$*
* gauge field: 
    * `GF` - *tower of gauge bilinears, $\mathcal{F}_{\mathcal X}^{(n)}$*

The model expects the following input:
* `phi`, `dphi` - *initial data on the inflaton, $\varphi$, $\dot\varphi$*
* `rhoChi` - *initial data on the fermion energy density, $\rho_{\chi}$*
* `beta` - *coupling strength, $\beta$*
* `V`, `dV` - *potential shape, $V(\varphi)$, $V_{,\varphi}(\varphi)$*

The model tracks the following events:
* end of inflation - terminate solver when $\ddot{a} < 0$
* negative energy - return an error when $\langle {\bf E}^2 \rangle$ or  $\langle {\bf B}^2 \rangle$ are negative 
"""
import numpy as np

from geff.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta, GF, BGVar
from geff.solver import TerminalEvent, ErrorEvent, GEFSolver
from geff.mbm import ModeSolver

from geff.utility.eom import (klein_gordon, friedmann, dlnkh_schwinger, ddelta, drhoChi, gauge_field_ode_schwinger,
                                        conductivities_collinear, conductivities_mixed, check_accelerated_expansion)
from geff.utility.boundary import boundary_approx_schwinger
from geff.utility.general import heaviside
from geff.utility.mode import mode_equation_SE_no_scale, damped_bd
from geff._docs import generate_docs, docs_models

name = "FAI basic"
"""The models name."""

settings = {"picture":"mixed"}
"""The model settings.

Possible settings are "mixed", "electric", "magnetic".

Determines if conductivities are computed assuming collinear E&M fields 
("electric", "magnetic") or not ("mixed").
"""

# parse settings
def define_conductivity():
    if settings["picture"]=="mixed":
        conductivity = conductivities_mixed
    elif settings["picture"]=="electric":
        def conductivity(a, H, E, B, G, omega):
            return conductivities_collinear(a, H, E, B, G, -1, omega)
    elif settings["picture"]=="magnetic":
        def conductivity(a, H, E, B, G, omega):
            return conductivities_collinear(a, H, E, B, G, 1, omega)
    else:
        raise KeyError(f"{settings['picture']} is an unknown choice for the setting'pic'")
    return np.vectorize(conductivity)

def interpret_settings():
    global conductivity
    conductivity = define_conductivity()
    return

#Define all additional variables
sigmaE=BGVar("sigmaE", 1, 0, "electric damping")
sigmaB=BGVar("sigmaB", 1, 0, "magnetic damping")
delta=BGVar("delta", 0, 0, "cumulative electric damping") 
xieff=BGVar("xieff", 0, 0, "effective instabilty parameter") 
s=BGVar("s", 0, 0, "electric damping parameter")
rhoChi=BGVar("rhoChi", 4, 0, "fermion energy density")

#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "time":[t],
            "dynamical":[N, phi, dphi, kh, delta, rhoChi],
            "static":[a, H, xi, E, B, G, ddphi, sigmaE, sigmaB, xieff, s],
            "constant":[beta],
            "function":[V, dV],
            "gauge":[GF]
            }

#State which variables require input for initialisation
model_input = [phi, dphi, rhoChi, beta, V, dV]

#this functions is called upon initialisation of the GEF class
def define_units(phi, dphi, V, rhoChi):
    #compute Hubble rate at t0
    rhoK = 0.5*dphi**2
    rhoV = V(phi)
    H0 = friedmann( rhoK, rhoV, rhoChi )
    
    omega = H0 #Characteristic frequency is the initial Hubble rate
    mu = 1. #Charatcterisic amplitude is the Planck mass
    
    return omega, mu

#the new function for sys_to_yini in GEFSolver
def initial_conditions(sys, ntr):
    yini = np.zeros((ntr+1)*3+6)

    #from the 'input' dictionary
    yini[0] = sys.N.value
    yini[1] = sys.phi.value
    yini[2] = sys.dphi.value

    #needs to be computed
    sys.initialise("kh")( abs(sys.dphi)*sys.beta )
    yini[3] = np.log(sys.kh.value)

    #initialise delta and rhoChi
    yini[4] = sys.delta.value
    yini[5] = sys.rhoChi.value

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

#define update_sys for GEFSolver
def update_values(t, y, sys, atol=1e-20, rtol=1e-6):
    #spacetime variables
    sys.t.value = t
    sys.N.value = y[0]
    sys.a.value = np.exp(y[0])

    #parse for convenience
    sys.phi.value = y[1]
    sys.dphi.value = y[2]
    sys.kh.value = np.exp(y[3])
    sys.delta.value = y[4]
    sys.rhoChi.value = y[5]

    #the gauge-field terms in y are not stored, save these values here
    sys.E.value = y[6]*np.exp(4*(y[3]-y[0]))
    sys.B.value = y[7]*np.exp(4*(y[3]-y[0]))
    sys.G.value = y[8]*np.exp(4*(y[3]-y[0]))

    #Hubble rate
    sys.H.value = ( friedmann(0.5*sys.dphi**2, sys.V(sys.phi), 
                                0.5*(sys.E+sys.B)*sys.omega**2, sys.rhoChi*sys.omega**2) )

    #conductivities
    sigmaE, sigmaB, ks = conductivity(sys.a.value, sys.H.value, sys.E.value,
                                       sys.B.value, sys.G.value, sys.omega)

    GlobalFerm = heaviside(np.log(ks),np.log(sys.a*sys.H))
    sys.sigmaE.value = (GlobalFerm*sigmaE)
    sys.sigmaB.value = (GlobalFerm*sigmaB)

    #boundary term parameters
    sys.s.value = sys.sigmaE/(2*sys.H)
    sys.xi.value = sys.beta*(sys.dphi/(2*sys.H))
    sys.xieff.value = sys.xi + sys.sigmaB/(2*sys.H)

    #acceleration for convenience
    sys.ddphi.value = klein_gordon(sys.dphi, sys.dV(sys.phi), sys.H, -sys.G*sys.beta*sys.omega**2)
    return

def compute_timestep(t, y, sys, atol=1e-20, rtol=1e-6):
    dydt = np.zeros(y.shape)

    #odes for N and phi
    dydt[0] = sys.H.value
    dydt[1] = sys.dphi.value
    dydt[2] = sys.ddphi.value

    #achieving dlnkhdt is monotonous requires some care
    dlnkhdt = dlnkh_schwinger( sys.kh, sys.dphi, sys.ddphi, sys.beta,
                                        0., sys.xieff, sys.s, sys.a, sys.H )
    xieff = sys.xieff.value
    s = sys.s.value
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= heaviside(dlnkhdt,0)*heaviside(logfc,y[3]*(1-1e-5))
    dydt[3] = dlnkhdt

    #the other derivatives are straight forwards
    dydt[4] = ddelta( sys.delta, sys.sigmaE )
    dydt[5] = drhoChi( sys.rhoChi, sys.E, sys.G, sys.sigmaE, sys.sigmaB, sys.H )

    #compute boundary terms and then the gauge-field bilinear ODEs
    Fcol = y[6:].shape[0]//3
    F = y[6:].reshape(Fcol,3)
    W = boundary_approx_schwinger(sys.xieff.value, sys.s.value)
    dFdt = gauge_field_ode_schwinger( F, sys.a, sys.kh, 2*sys.H*sys.xieff,
                                            sys.sigmaE, sys.delta,
                                                W, dlnkhdt )
    #reshape to fit dydt
    dydt[6:] = dFdt.reshape(Fcol*3)

    return dydt

#Event 1: Track the end of inflation:
def condition_EndOfInflation(t, y, sys):
   #compute energy densities
    rhoV = sys.V(y[1])
    rhoK = y[2]**2/2
    rhoEM = 0.5*(y[6]+y[7])*(sys.omega/sys.mu)**2*np.exp(4*(y[3]-y[0]))
    rhoF = y[5]*(sys.omega/sys.mu)**2

    #compute pressure
    pV = -rhoV
    pK = rhoK
    pEM = 1/3*rhoEM
    pF = 1/3*rhoF

    return check_accelerated_expansion([rhoV, rhoK, rhoEM, rhoF], [pV, pK, pEM, pF])/(sys.H)**2

def consequence_EndOfInflation(sys, occurance):
    if occurance:
        #stop solving once the end of inflation is reached
        return "finish", {}
    else:
        #increase tend given the current Hubble rate
        tdiff = np.round(5/sys.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(sys.t + tdiff, 0)
        return "proceed", {"tend":tend}
    
EndOfInflation = TerminalEvent("End of inflation", condition_EndOfInflation, -1, consequence_EndOfInflation)
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
MbM = ModeSolver(mode_equation_SE_no_scale, {"a":a, "xieff":xieff, "H":H, "sigmaE":sigmaE},
                         damped_bd, {"a":a, "sigmaE":sigmaE, "delta":delta}, default_atol=1e-5)
"""The mode solver used by the GEF model."""


#define default docs for the above functions
generate_docs(docs_models.DOCS)