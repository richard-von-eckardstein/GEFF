r"""
$\newcommand{\bm}[1]{\boldsymbol{#1}}$
Defines the GEF model "fai_kh" corresponding to fermionic axion inflation with a heuristic scale dependence model through the instability scale $k_{\rm h}$.

For more details on this model, see e.g., [2408.16538](https://arxiv.org/abs/2408.16538).

---

The model knows the following variables:
* time variable: `t` - *cosmic time*, $t$ 
* dynamical variables:
    * `N` - *$e$-folds*,  $N$
    * `phi`, `dphi` - *inflaton amplitude, $\varphi$, and velocity, $\dot{\varphi}$* 
    * `kh` -  *the instability scale, $k_{\rm h}$*
    * `rhoChi` - *fermion energy density, $\rho_{\chi}$*
* static variables:
    * `a` - *scale factor, $a$* 
    * `H` - *Hubble rate, $H$* 
    * `ddphi` - *inflaton acceleration, $\ddot{\varphi}$*
    * `E`, `B`, `G` - *gauge-field expectation values, $\langle \bm{E}^2 \rangle$, $\langle \bm{B}^2 \rangle$, -$\langle \bm{E} \cdot \bm{B} \rangle$*
    * `xi` - *instability parameter, $\xi$* 
    * `sigmaE`, `sigmaB` - *electric and magnetic conductivities, $\sigma_{\rm E}$, $\sigma_{\rm B}$*
    * `xieff` - *effective instability parameter, $\xi_{\rm eff}$*
    * `kS` - *fermion momentum scale, $k_{\rm S}$*
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
* negative norms - return an error when $\langle \bm{E}^2 \rangle$ or  $\langle \bm{B}^2 \rangle$ are negative 
"""
import numpy as np

from geff.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta, GF, define_var
from geff.solver import TerminalEvent, ErrorEvent, GEFSolver
from geff.mbm import ModeSolver

from geff.utility.eom import (klein_gordon, friedmann, dlnkh, drhoChi, gauge_field_ode_schwinger,
                                        conductivities_collinear, conductivities_mixed, check_accelerated_expansion)
from geff.utility.boundary import boundary_pai
from geff.utility.general import heaviside
from geff.utility.mode  import bd_classic, mode_equation_SE_scale
from geff._docs import generate_docs, docs_models

name = "FAI kh"
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
        raise KeyError(f"{settings['picture']} is an unknown choice for the setting'picture'")
    return np.vectorize(conductivity)

def interpret_settings():
    global conductivity
    conductivity = define_conductivity()
    return

#Define additional variables
sigmaE=define_var("sigmaE", 1, 0, "electric damping")
sigmaB=define_var("sigmaB", 1, 0, "magnetic damping")
xieff=define_var("xieff", 0, 0, "effective instability parameter")
rhoChi=define_var("rhoChi", 4, 0, "fermion energy density")
kS=define_var("kS", 1, 0, "fermion momentum scale")#Fermion energy density 

#Assign quantities to a dictionary, classifying them by their role:
quantities={
            "time":[t], #time coordinate according to which EoMs are expressed
            "dynamical":[N, phi, dphi, kh, rhoChi], #variables which evolve in time according to an EoM
            "static":[a, H, xi, E, B, G, ddphi, sigmaE, sigmaB, xieff, kS], #variables which are derived from dynamical variables
            "constant":[beta], #constant quantities in the model
            "function":[V, dV], #functions of variables such as scalar potentials
            "gauge":[GF] #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
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
    yini = np.zeros((ntr+1)*3+5)

    #from the 'input' dictionary
    yini[0] = sys.N.value
    yini[1] = sys.phi.value
    yini[2] = sys.dphi.value

    #needs to be computed
    sys.initialise("kh")( abs(sys.dphi)*sys.beta )
    yini[3] = np.log(sys.kh.value)

    #initialise rhoChi
    yini[4] = sys.rhoChi.value

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
    
    sys.rhoChi.value = y[4]

    #the gauge-field terms in y are not stored, save these values here
    sys.E.value = y[5]*np.exp(4*(y[3]-y[0]))
    sys.B.value = y[6]*np.exp(4*(y[3]-y[0]))
    sys.G.value = y[7]*np.exp(4*(y[3]-y[0]))

    #Hubble rate
    sys.H.value = friedmann(0.5*sys.dphi**2, sys.V(sys.phi), 
                                0.5*(sys.E+sys.B)*sys.omega**2, sys.rhoChi*sys.omega**2)
    
    #conductivities
    sigmaE, sigmaB, ks = conductivity(sys.a.value, sys.H.value, sys.E.value,
                                       sys.B.value, sys.G.value, sys.omega) 
    sys.kS.value = ks
    GlobalFerm = heaviside(np.log(ks), np.log(sys.a*sys.H))
    sys.sigmaE.value = GlobalFerm*sigmaE
    sys.sigmaB.value = GlobalFerm*sigmaB

    #boundary term parameters
    sys.xi.value = sys.beta*(sys.dphi/(2*sys.H))
    sys.xieff.value = sys.xi + sys.sigmaB/(2*sys.H)

    #acceleration for convenience
    sys.ddphi.value = klein_gordon(sys.dphi, sys.dV(sys.phi),  sys.H, -sys.G*sys.beta*sys.omega**2)
    return

def compute_timestep(t, y, sys, atol=1e-20, rtol=1e-6):
    dydt = np.zeros(y.shape)

    #odes for N and phi
    dydt[0] = sys.H.value
    dydt[1] = sys.dphi.value
    dydt[2] = sys.ddphi.value

    #achieving dlnkhdt is monotonous requires some care
    dlnkhdt = dlnkh( sys.kh, sys.dphi, sys.ddphi, sys.beta,
                       0., sys.xi, sys.a, sys.H )
    r = 2*abs(sys.xi)
    logfc = y[0] + np.log(r*dydt[0]) 
    dlnkhdt *= heaviside(dlnkhdt, 0)*heaviside(logfc, y[3]*(1-1e-5))
    dydt[3] = dlnkhdt

    #ode for rhoChi
    dydt[4] = drhoChi( sys.rhoChi, sys.E, sys.G,
                         sys.sigmaE, sys.sigmaB, sys.H )

    #compute boundary terms and then the gauge-field bilinear ODEs
    Fcol = y[5:].shape[0]//3
    F = y[5:].reshape(Fcol,3)
    W = boundary_pai(float(sys.xi.value))
    dFdt = gauge_field_ode_schwinger( F, sys.a, sys.kh, 2*sys.H*sys.xieff,
                    sys.sigmaE, 1.0, W, dlnkhdt )
    #reshape to fit dydt
    dydt[5:] = dFdt.reshape(Fcol*3)

    return dydt


#Event 1: Track the end of inflation:
def condition_EndOfInflation(t, y, sys):
    #compute energy densities
    rhoV = sys.V(y[1])
    rhoK = y[2]**2/2
    rhoEM = 0.5*(y[5]+y[6])*(sys.omega/sys.mu)**2*np.exp(4*(y[3]-y[0]))
    rhoF = y[4]*(sys.omega/sys.mu)**2

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

#Event 2: ensure that E^2 and B^2 are positive
def condition_NegativeNorms(t, y, sys):
    return min(y[5], y[6])
    
NegativeNorms : ErrorEvent = ErrorEvent("Negative norms", condition_NegativeNorms, -1, "Negative value for E^2 or B^2.")
"""Defines the 'Negative norms' event."""

events = [EndOfInflation, NegativeNorms]


#gather all information in the solver
solver = GEFSolver(initial_conditions, update_values, compute_timestep, quantities, events)
"""The solver used by the GEF model."""

#define mode-by-mode solver
MbM = ModeSolver(mode_equation_SE_scale, {"a":a, "xi":xi, "H":H, "sigmaE":sigmaE, "sigmaB":sigmaB, "kS":kh},
                         bd_classic, {}, default_atol=1e-5)
"""The mode solver used by the GEF model."""


#define default docs for the above functions
generate_docs(docs_models.DOCS)