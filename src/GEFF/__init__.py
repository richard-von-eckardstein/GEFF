r"""
Welcome to our visitors tour of the **Gradient Expansion Formalism Factory**!

# GEFF: Established in 2025

Use this python package to investigate gauge-field production during cosmic inflation.

If you are interested in axion inflation, the package comes with everything you need:
- **pre defined flavours**
    - pure axion inflation (PAI) : *good old axion inflation*
    - fermionic axion inflation (FAI) : *coupled to Standard Model fermions!*
- **useful tools**
    - Resolve the dynamics of axion inflation including homogeneous backreaction.
    - Investigate the gauge-field spectrum.
    - Determine the vacuum and induced tensor power spectrum.
    - Compute gravitational wave spectra.


But we don't want to hold you back! The package provides a flexible framework to create your **own GEF flavour**, with all tools at your disposable. 
It is indeed a true GEF *factory* !

# The refreshing taste of GEF

The *gradient expansion formalism* (or GEF) is an elegant numerical technique to determine the dynamics and backreaction of gauge-fields during inflation
by directly evolving the time-dependent quantum expectation values of the gauge-fields, 
e.g., $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, $\langle {\bf E} \cdot {\bf B} \rangle$ etc.
If this is the first time you encounter the GEF, here are some useful articles on the topic:
* ...
* ...

Summarized, the strategy is to take the (charge-free) Maxwell's equations in an expanding spacetime,

<a name="max">$$\operatorname{div} {\bf E} = 0\, , \qquad \operatorname{div} {\bf B} = 0\, ,$$</a>
$$\dot{{\bf E}} + 2 H {\bf E} - \frac{1}{a}\operatorname{rot} {\bf B} + {\bf J} = 0 \, ,$$
$$\dot{{\bf B}}  + 2 H {\bf B} + \frac{1}{a}\operatorname{rot} {\bf E} = 0 \,$$
and reformulate them into an infinite tower of ODE's for the variables

$$ \mathcal{F}_{E}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm UV}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\, ,$$
$$ \mathcal{F}_{G}^{(n)} = -\frac{a^4}{2 k_{{\rm UV}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf B} + {\bf B} \cdot \operatorname{rot}^n {\bf E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)} \frac{{\rm d} k}{k} \frac{a k^{n+4}}{2 \pi^2 k_{{\rm UV}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)] \, ,$$
$$ \mathcal{F}_{B}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm UV}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2 \, ,$$

for a suitably chosen regularization scale $k_{\rm UV}$. These ODE's are then given by

$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)} + 2 \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf E} \rangle =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right) + \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf B} \rangle= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

which may be truncated at some order $n_{\rm tr}$ by an analytical closing condition. The source functions $S_{\mathcal{X}}^{(n)}$ account for the time variation of the UV regulator $k_{\rm UV}$.

These quantities may then be coupled to other equations to determine the backreaction of gauge-fields on the inflationary dynamics.
Once the dynamical background has been computed in this manner, it can be used to determine all manner of interesting results, e.g., the tensor power spectrum induced by gauge fields.

# Sampling the GEF flavours

## Choosing your flavour

We start by sampling the GEF flavour of our choice. Let's try "classic" (corresponds to PAI):

```python
from GEFF import GEF

# Create the GEF model of your choice
ClassicGEF = GEF("classic")
``` 
Next up, we initialize `ClassicGEF`. It expects information on the inflaton--gauge-field coupling strength $\beta / M_{\rm P}$, 
initial conditions for the inflaton field, $\varphi(0)$, $\dot\varphi(0)$, and the shape of the inflaton potential, $V(\varphi)$.
Let's configure the model to start on the slow-roll attractor of a chaotic inflation potential:

$$\beta = 15,\, \qquad  m = 6.16 \times 10^{-6} M_{\rm P}$$
$$ \varphi(0) = 15.55 M_{\rm P}, \qquad \dot{\varphi}(0) = -\sqrt{\frac{2}{3}} m M_{\rm P}, \qquad V(\varphi) = \frac{1}{2}m^2 \varphi^2$$

We can pass all the required information in the form of three dictionaries:
```python
import numpy as np

m = 6.16e-6
beta = 15
phi = 15.55
dphi = -np.sqrt(2/3)*m

def V(x): return 0.5*m**2*x**2
def dV(x): return m**2*x

G = ClassicGEF({"beta":beta}, {"phi":phi, "dphi":dphi}, {"V":V, "dV":dV})
```
If you want to know what input is expected by the GEF model, use the `print_input` method of the model.

> **A note on units**:
> We work in Planck units $M_{\rm P}=1$, which sets the energy scale used by the GEF model. 
> The model will also set an inverse-time scale, the Hubble rate at initialization, $H_0 \sim m$.
> Internally, the code works with dimensionless quantities, e.g., $\bar{X} = X H_0^{-a} M_{\rm P}^{-b}$ with $a$ and $b$ indicating
> how $X$ scales with inverse time and energy. For example, the inflaton velocity scales like $\dot{\varphi} = \dot{\bar{\varphi}}H_0 M_{\rm P}$,
> i.e., like a scalar amplitude, $\varphi \sim M_{\rm P}$, and a time derivative, $\partial_t \sim H_0$.
> Throughout the code, we refer to the dimensionless variable, $\bar{X}$, as being in "numerical units", while $X$ is in "physical units". 
> Don't worry, this is happening under the hood, but if you want more details, see `.bgtypes`.

## Getting a taste

Now, that our model is initialized, we can start our first GEF run:
```python
sol, spec = G.run()
```
The output `sol` contains the solution obtained by the GEF model and some useful statistics, but is only a byproduct. All the
information about our dynamical system have already been moved to the object `G`:
```python
import matplotlib.pyplot as plt
# Plot the evolution of the inflaton amplitude as a function of e-folds:
plt.plot(G.N, G.phi)
plt.show()
``` 
How did we know the GEF model has the attributes `N` and `phi`? You can find out using `G.value_names()`.
If you want to know all variables the GEF knows about, use `G.quantity_names()`.
You can get a brief description of the variable `X`, by using `X.what`.

> **A note on variables:**
> The variables encoded in a GEF model are a custom class object called `Val`. It works like a numpy array. 
> The potential which we defined earlier has also been converted to a custom class called `Func`. You can use it like a regular function,
> but it is best to call it with `Val` objects.
> These classes are defined to take care of unit conversions. If you are curious, have a look at `.bgtypes`.

We did not only get back a `sol` object, but also `spec`. This object contains the gauge-field mode functions $A_\lambda(t,k)$.
The object `spec` is an instance of the `GaugeSpec` class, which defines useful methods for handling time-dependent gauge-field spectra.
For more information, see `.mbm.GaugeSpec`.

Next, let us store our GEF and mode solutions in a file:
```python
# some dummy paths for illustration
gefpath = "some_gef_file.dat"
mbmpath = "some_mbm_file.dat" 
G.save_GEFdata(gefpath)
spec.save_spec(mbmpath)
```

> **A note on storage:**
> All quantities are stored as dimensionless variables by the GEF code.
> We therefore recommend you use the GEF class to also load the data, `G.load_GEFdata()`.
> This ensures you are using the same reference scale $H_0$ to restore dimensions.

## A rich palette

That's not all the GEF object `G` can do. For example, let's use it to compute the tensor power spectrum:
```python
from GEFF.tools.pt import PT

# Use the GEF's knowledge of the background expansion
# to compute the vacuum and induced power spectrum for 100 momentum modes k:
ks, pt_dic =  PT(G).compute_pt(100, mbmpath)
``` 
We can just pass `G` to `PT` and all relevant information for the computation is extracted. It's that easy!

To finish out this tutorial, let us sample a second GEF flavour, "SE_kh":
```python
FermionGEF = GEF("SE_kh", {"picture":"electric"})
...
```
This model comes in three varieties ("pictures") corresponding to the effective treatment of fermions in the model.
We specified the particular variety py passing a `settings` dictionary upon creating the model.

You can get all GEF flavours at your disposal by using `GEFF.take_a_tour`.
 
# Create your own flavour

Having explored the potential of the GEFF code, you may be inclined to make your own GEF flavour.
To help you along, we will guide you through this process for an example toy model.

## The first step is the hardest

First, we need to work out the mathematical formulation of our model.

Let us consider the case of Abelian gauge-field production in de Sitter space by a current of the type ${\bf J} = \xi(t) {\bf B}$.
The ODE tower for the gauge-field bilinears are then neatly closed:
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} + 2 \frac{\dot{f}}{f}\right] \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm h}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} +\frac{\dot{f}}{f}\right]  \mathcal{F}_{G}^{(n)} - \frac{k_{\rm h}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right)= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm h}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$
One can determine that a sensible regularization scale for this model is given by $k_{\rm h}(t) = 2aH \underset{s \leq t}{\max}(|\xi(s)|)$.

Once we have worked out the equations, we can turn to writing a GEF model file.

## Only the best ingredients

The first thing we should state in our model file is the name of the new GEF flavour:
```python
import numpy as np

name = "tutorial"
```
> **A note on settings**: Following the model's name, you can also define a `settings` dictionary.
> This dictionary should contain some default key&ndash;value pair defining a setting name and its default setting.
> The `settings` should be accompanied by a function called `interpret_settings`. This method will be called upon creating the model using the `GEF` factory,
> so you can use it to tell the model how to handle user input settings. See e.g. `.models.SE_kh` for how this is done in practice.

Next, we need to define define and categorize the variables which appear in our model. This is taken care of by the functions `.bgtypes.BGVal` and `.bgtypes.BGFunc`.
The `bgtypes` module also contains some pre-defined variables, which are often encountered in GEF models.

There are three variables which every GEF model needs to define:
* **cosmic time $t$**: the solver is solving all ODE's in terms of $t$ starting from $t=0$.
* **$e$-folds $N$**: needed by most internal methods. It should be defined such that $N(t=0)=0$.
* **Hubble rate $H$**: needed by most internal methods.

Beyond these staples, some extra variables appear in our ODE's:
* $\mathcal{F}_{X}^{(n)}$: the gauge-field bilinears.
* $\xi(t)$: the instability parameter as a function of time.
* $k_{\rm h}(t)$: the UV regulator.
* $\dot{\xi}$: needed to compute ${\rm d} \ln k_{\rm h} / {\rm d}t$.

To properly account for the gauge-fields, we also need to add in the following three variables:
* $\mathcal{E}^{(0)} = \langle {\bf E}^2 \rangle$, *called "E" by the GEF*
* $\mathcal{B}^{(0)} = \langle {\bf B}^2 \rangle$, *called "B" by the GEF*
* $\mathcal{G}^{(0)} = -\langle {\bf E} \cdot {\bf B} \rangle$, *called "G" by the GEF*

> **Note on gauge field bilinears** The time evolution of the variables $\mathcal{F}_{X}^{(n)}$ will not be saved by the GEFF code,
> since we are typically only interest in the quantities with $n=0$. The output `sol` returned by the `run` method contains the full information on the $\mathcal{F}_{X}^{(n)}$,
> but only the information on $n=0$ is stored in the GEF object in the form of  `E`, `B` and `G`

All these variables are defined in our model file as follows:
```python
from GEFF.bgtypes import BGVal, BGFunc

# We make use of the fact that a lot of these variables are pre-defined:
from GEFF.bgtypes import t, N, H, a, E, B, G, kh

# We also need to define some new objects:
xi = BGFunc("xi", [t], H0=0, MP=0) # dimensionless function of t
dxi = BGFunc("xi", [t], H0=0, MP=0) # dimensionless function of t

# lastly, we give a name to the gauge-field variable F_X^n:
GF = type("GF", (object,), {"name":"GF","0thOrder":{E, B, G}, "UV":kh})
```
The new quantities `xi` and `dxi` are defined as `Func` objects with argument `t` using the class factory `BGFunc`.
However, the gauge-field variable `GF` is handled differently. In order for the GEFF package to work properly, we needed to tell it
that the objects `E`, `B`, and `G` are related to it via $\mathcal{X}^{(0)} = (k_{\rm h}/a)^4\mathcal{F}_{\mathcal{X}}^{(0)}$
with the regularization scale $k_{\rm h}$ represented by the object `kh`.

All the variables we have defined serve a specific purpose in our GEF model. To inform the GEFF of this, we need to classify each of them in one of these categories:
* **time**: a time variable (needs to be `t`)
* **dynamical**: variables that evolves with time according to some ODE solved by the GEF.
* **gauge**: the gauge-field variable `GF`
* **static**: variables that are computed through other variables (e.g, "time", "dynamical" or "gauge").
* **constant**: variables representing constants of time.
* **function**: functions of the above.

In our case, this would look as follows:
```python
quantities = {
            "time":{t}, # this is mandatory!
            "dynamical":{kh}, # kh is best evolved from an ODE
            "gauge":{GF}, # state the obvious
            "static":{N, a, E, B, G}}, # directly computed from other variables
            "constant":{H}, # we work in perfect de-Sitter space
            "function":{xi, dxi} # a priori undetermined functions of t
            }
```

## Write a recipe

With the variables defined, an important step towards our GEF model is already taken. Next, let us set up the differential equation solver.

Internally, the GEFF package uses `scipy.integrate.solve_ivp` to solve differential equations. 
This requires that ODE's are formulated as $\dot{\vec{y}} = f(t, \vec{y})$ with $\vec{y}$ as a `numpy` array.
However, the GEFF prefers the `BGSystem` class, which takes care of unit conversions.
So, we need to define how to translate between the two.

First up, we define how to interpret user input to initialize the array $\vec{y}$:
We can assume that all our input is stored in `sys`, and we should return an array:
```python
def initial_conditions(sys, ntr):
    yini = np.zeros(1 + 3*(ntr+1))
    yini[0] = np.log(2*sys.xi(0)*sys.H) #index 0 is log(kh)
    # initialise all F_X^n as zero at indices [1:]

    return yini
``` 
Importantly, upon defining `initial_conditions`, we also make a choice; which entries in $\vec{y}$ correspond to which dynamical variable.
In our simple toy model, there actually is no choice to be made: The GEFF package expects that the gauge-field variables are stored *after* all
dynamical variables, as the number of the $\mathcal{F}_\mathcal{X}^{(n)}$'s will vary depending on $n_{\rm tr}$.
Hence, $k_{\rm h}$ necessarily goes first. However, we do have the choice of evolving $\log k_{\rm h}$ instead of $k_{\rm h}$.

Next, we write the recipe used to define our GEF ODE. The ODE evolution is computed in two steps:
1. `update_values`: update `sys` according to $\vec{y}(t)$ and $t$
2. `timestep`: compute $\dot{\vec{y}}(t)$ from `sys`

To define `update_values`, we can use that `sys` knows all the variables which we have previously declared:
```python
# Note how we can use the Val objects like arrays, and the Func objects like functions
def update_values(t, y, sys, atol, rtol):
    # evolution of spacetime
    sys.t.set_value(t)
    sys.N.set_value(sys.t*sys.H) #perfect de Sitter
    sys.a.set_value(np.exp(sys.N))

    # define how kh is computed from xi:
    sys.kh.set_value( np.exp(y[0] ) # y[0] is log(kh)

    # use that y[1] = F_E^0, y[2] = F_B^0, y[3] = F_G^0
    sys.E.set_value( (sys.kh/sys.a)**4 * y[1] )
    sys.B.set_value( (sys.kh/sys.a)**4 * y[2] )
    sys.G.set_value( (sys.kh/sys.a)**4 * y[3] )
    return
```
For `timestep`, we can make use of some pre-defined functions in the `.utility` module.

```python
from GEFF.utility.aux_eom import gauge_field_ode
from GEFF.utility.boundary import boundary_approx
from GEFF.utility.auxiliary import heaviside

def timestep(t, y, sys, atol, rtol):
    dydt = np.zeros_like(y)

    xi_t = sys.xi(sys.t)

    dlnkhdt = sys.dxi(sys.t)/abs(xi_t) #first guess for derivative
    logfc = sys.N + np.log( 2*abs(sys.xi(t))*sys.H) #non-monotonic kh
    # ensure monotonicity using heaviside functions
    eps = max(abs(y[0])*rtol, atol) #tolerance prameter used by heaviside
    dlnkhdt *= heaviside(dlnkhdt, eps)*heaviside(logfc-y[3]+10*eps, eps)
    dydt[0] = dlnkhdt

     #compute boundary terms
    W = boundary_approx(xi_t)

    # reshape arrays to fit gauge_field_ode
    Fcol = y[1:].shape[0]//3
    F = y[1:].reshape(Fcol,3)
    
    # compute the gauge-field ODEs
    dFdt = gauge_field_ode(F, sys.a, sys.kh, 2*sys.H*sys.xi_t, W, dlnkhdt)
    # note that we can use 'a', 'kh' etc.
    # 'update_values' is always called before 'timestep'
    #reshape to fit dydt
    dydt[1:] = dFdt.reshape(Fcol*3)

    return dydt
``` 
> **Note on tolerance parameters:**`atol`and `rtol` in `update_values` and `timestep` refer to the absolute and relative tolerances used by `solve_ivp` 
> to solve differential equations. It is best to account for these tolerance when computing heaviside functions, as we have done in the example above.

These are all the ingredients we need to solve the GEF ODE's. We can combine them using the  `.solver.GEFSolver` class factory:

```python
# pass everything we have defined to GEFSolver.
solver = GEFSolver(initial_conditions, update_values, compute_timestep, quantities)
``` 

We also should define how to compute gauge-spectra from a GEF solution. In this toy model,
we can just use a pre-defined class. For more advanced cases, see `.mbm.ModeSolver`.
```python
from GEFF.mbm import BaseModeSolver

MbM = BaseModeSolver
```

## The finishing touch

All the ingredients of our GEF model are in `solver` and `MbM`. The last thing we need to do, is define how our new GEF model is initialized.
First, we need to declare, what input our GEF model expects from the user.
Our simple GEF model initializes all gauge-fields at zero, and $k_h$ is determined from $t$ and $\xi$. So `initial data` expects nothing.
The user does need to input the Hubble rate, $H$. It should be passed as a constant.
Finally, the the instability function $\xi(t)$, and its derivative, $\dot{\xi}$, need to be given on initialization,
 similarly to how the "classic" model needed $V$ and $V_{,\varphi}$ as input. Our model file therefore states 
```python
input = {
        "initial data":{},
        "constants":{"H"},
        "functions":{"xi", "dxi"}
        }
``` 
The last step is to define the units of our GEF model based on the user input. You can do this through the `define_units` function:
```python
def define_units(consts, init, funcs):
    # The characteristic inverse time scale is the constant Hubble rate in Planck units
    freq = consts["H"] 
    # The charateristic energy scale is the Planck mass (in Planck units)
    amp = 1. 
    return freq, amp
```
We are finally done! We can put everything together in a file, let's call it "tutorial.py", and we are good to go!
If all went well, you can now use your own GEF flavour just like the pre-defined ones:
```python
import numpy as np
from GEFF import GEF

TutorialGEF = GEF("tutorial")

H = 5e-6
def xi(x): return 5*(np.sin(np.pi*x/5)+1) 
def dxi(x): return np.pi*np.cos(np.pi*x/5)

G = TutorialGEF({"H":H}, {}, {"xi":xi, "dxi":dxi})

G.run()
...
```





"""



from .gef import GEF, BaseGEF
"""from GEFF import gef, bgtypes, mbm, models, tools, solver, utility

__all__ = [gef, bgtypes, mode_by_mode, models, tools, solver, utility]"""
__version__ = "0.1"

def take_a_tour():
    """**TODO**"""
    pass


