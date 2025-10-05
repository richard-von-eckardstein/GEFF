r"""
Defines the GEF model "classic" corresponding to pure axion inflation.

For more details on this model, see e.g., [arXiv:2109.01651](https://arxiv.org/abs/2109.01651).

---

The model knows the following variables:
* time variable: `t` - *cosmic time*, $t$ 
* dynamical variables:
    * `N` - *$e$-folds*,  $N$
    * `phi`, `dphi` - *inflaton amplitude, $\varphi$, and velocity, $\dot{\varphi}$* 
    * `kh` -  *the instability scale, $k_{\rm h}$*
* static variables:
    * `a` - *scale factor, $a$* 
    * `H` - *Hubble rate, $H$* 
    * `ddphi` - *inflaton acceleration, $\ddot{\varphi}$*
    * `E`, `B`, `G` - *gauge-field expectation values, $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, -$\langle {\bf E} \cdot {\bf B} \rangle$*
    * `xi` - *instability parameter, $\xi$* 
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

from GEFF.bgtypes import t, N, a, H, phi, dphi, ddphi, V, dV, E, B, G, xi, kh, beta, GF
from GEFF.solver import TerminalEvent, ErrorEvent, GEFSolver
from GEFF.mbm import ModeSolver

from GEFF.utility.eom import friedmann, gauge_field_ode, dlnkh, klein_gordon, check_accelerated_expansion
from GEFF.utility.mode import bd_classic, mode_equation_classic
from GEFF.utility.boundary import boundary_approx
from GEFF.utility.general import heaviside
from GEFF._docs import generate_docs, docs_models


name : str = "classic"
"""The models name."""

quantities : dict={
            "time":[t], #time coordinate according to which EoMs are expressed
            "dynamical":[N, phi, dphi, kh], #variables which evolve in time according to an EoM
            "static":[a, H, ddphi, xi, E, B, G], #variables which are derived from dynamical variables
            "constant":[beta], #constant quantities in the model
            "function":[V, dV], #functions of variables such as scalar potentials
            "gauge":[GF] #Gauge fields whose dynamics is given in terms of bilinear towers of expectation values
            }

#State which variables require input for initialisation
input_dic = [ beta, phi, dphi, V, dV]

#this functions is called upon initialisation of the GEF class
def define_units(phi, dphi, V):
    #compute Hubble rate at t0
    rhoK = 0.5*dphi**2
    rhoV = V(phi)
    H0 = friedmann( rhoK, rhoV )
    
    omega = H0 #Characteristic frequency is the initial Hubble rate in Planck units
    mu = 1. #Charatcterisic amplitude is the Planck mass (in Planck units)

    return omega, mu

#define sys_to_yini for GEFSolver
def initial_conditions(sys, ntr):
    yini = np.zeros((ntr+1)*3+4)

    #from the 'input' dictionary
    yini[0] = sys.N.value
    yini[1] = sys.phi.value
    yini[2] = sys.dphi.value

    #needs to be computed
    sys.initialise("kh")( abs(sys.dphi)*sys.beta )

    #all gauge-field expectation values are assumed to be 0 at initialisation
    return yini

#define update_sys for GEFSolver
def update_values(t, y, sys):
    #spacetime variables
    sys.t.value = t
    sys.N.value = y[0]
    sys.a.value = np.exp(y[0])

    #parse for convenience
    sys.phi.value = y[1]
    sys.dphi.value =  y[2]
    sys.kh.value = np.exp(y[3])

    #the gauge-field terms in y are not stored, save these values here
    sys.E.value = y[4]*np.exp(4*(y[3]-y[0]))
    sys.B.value = y[5]*np.exp(4*(y[3]-y[0]))
    sys.G.value = y[6]*np.exp(4*(y[3]-y[0]))

    #Hubble rate
    sys.H.value = friedmann(0.5*sys.dphi**2, sys.V(sys.phi), 0.5*(sys.E+sys.B)*sys.omega**2)

    #boundary term parameter
    sys.xi.value = sys.beta*(sys.dphi/(2*sys.H))

    #acceleration for convenience
    sys.ddphi.value = klein_gordon(sys.dphi, sys.dV(sys.phi), sys.H, -sys.G*sys.beta*sys.omega**2)
    return

#define timestep for GEFSolver
def compute_timestep(t, y, sys):

    dydt = np.zeros(y.shape)

    #odes for N and phi
    dydt[0] = sys.H.value
    dydt[1] = sys.dphi.value
    dydt[2] = sys.ddphi.value

    #achieving dlnkhdt is monotonous requires some care
    dlnkhdt = dlnkh( sys.kh, sys.dphi, sys.ddphi, sys.beta,
                       0., sys.xi, sys.a, sys.H )
    
    logfc = y[0] + np.log( 2*abs(sys.xi)*dydt[0])
    dlnkhdt *= heaviside(dlnkhdt,0)*heaviside(logfc,y[3]*(1-1e-5))
    dydt[3] = dlnkhdt

    #compute boundary terms and then the gauge-field bilinear ODEs
    Fcol = y[4:].shape[0]//3
    F = y[4:].reshape(Fcol,3)
    W = boundary_approx(sys.xi.value)
    dFdt = gauge_field_ode(F, sys.a, sys.kh, 2*sys.H*sys.xi, W, dlnkhdt, L=20)

    #reshape to fit dydt
    dydt[4:] = dFdt.reshape(Fcol*3)

    return dydt

#Event 1: Track the end of inflation:
def condition_EndOfInflation(t, y, sys):
    #compute energy densities
    rhoV = sys.V(y[1])
    rhoK = y[2]**2/2
    rhoEM = 0.5*(y[4]+y[5])*(sys.omega/sys.mu)**2*np.exp(4*(y[3]-y[0]))

    #compute pressure
    pV = -rhoV
    pK = rhoK
    pEM = 1/3*rhoEM

    return check_accelerated_expansion([rhoV, rhoK, rhoEM], [pV, pK, pEM])/(sys.H)**2

def consequence_EndOfInflation(sys, occurrence):
    if occurrence:
        #stop solving once the end of inflation is reached
        return "finish", {}
    else:
        #increase tend given the current Hubble rate
        tdiff = np.round(5/sys.H, 0)
        #round again, sometimes floats cause problems in t_span and t_eval.
        tend  = np.round(sys.t + tdiff, 0)
        return "proceed", {"tend":tend}

EndOfInflation : TerminalEvent = TerminalEvent("End of inflation", condition_EndOfInflation, -1, consequence_EndOfInflation)
"""Defines the 'End of inflation' event."""

#Event 2: ensure energy densities that are positive definite do not become negative
def condition_NegativeEnergies(t, y, sys):
    return min(y[4], y[5])
    
NegativeEnergies : ErrorEvent = ErrorEvent("Negative energies", condition_NegativeEnergies, -1)
"""Defines the 'Negative energy' event."""

events = [EndOfInflation, NegativeEnergies]

#gather all information in the solver
solver = GEFSolver(initial_conditions, update_values, compute_timestep, quantities, events)
"""The solver used by the GEF model."""

#define mode-by-mode solver
MbM = ModeSolver(mode_equation_classic, ["a","xi", "H"], bd_classic, [])
"""The mode solver used by the GEF model."""


#define default docs for the above functions
generate_docs(docs_models.DOCS)





