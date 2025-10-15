r"""
$\newcommand{\bm}[1]{\boldsymbol{#1}}$

Welcome to our visitors tour of the **Gradient Expansion Formalism Factory**!

---

# GEFF: Established in 2025

This python package is designed to handle gauge-field production during cosmic inflation
using the *gradient expansion formalism* (GEF).

If you are interested in axion inflation, the package comes with everything you need:
- **a variety of flavors**
    - pure axion inflation (PAI) : *good ol' axion inflation*
    - fermionic axion inflation (FAI) : *axion inflation with Standard Model fermions!*
- **useful tools**
    - Resolve the dynamics of axion inflation including homogeneous backreaction.
    - Analyze the gauge-field spectrum.
    - Determine the vacuum and induced tensor power spectrum.
    - Compute gravitational-wave spectra.


But we don't want to hold you back! The package provides a flexible framework to create your **own GEF flavor**, with all built-in tools at your disposable. 
It is indeed a true GEF *factory* !

You can install this package using pip

```bash
pip install cosmo-geff
```

or using the `geff.yml` file found at the [GitHub repository](https://github.com/richard-von-eckardstein/GEFF) for this package,

```bash
conda env create -f geff.yml
``` 

---

# The refreshing taste of GEF

The GEF is a numerical technique to determine the dynamics and backreaction of gauge-fields during inflation
by directly evolving the time-dependent quantum expectation values of the gauge field, 
e.g., $\langle \bm{E}^2 \rangle$, $\langle \bm{B}^2 \rangle$, $\langle \bm{E} \cdot \bm{B} \rangle$, etc.
If this is the first time you encounter the GEF, here are some useful articles on the topic:
* [2109.01651](https://arxiv.org/abs/2109.01651)
* [2310.09186](https://arxiv.org/abs/2310.09186)
* [2408.16538](https://arxiv.org/abs/2408.16538)

The strategy behind the GEF is to take Maxwell's equations in an expanding spacetime,

<a name="max">$$\operatorname{div} \bm{E} = 0\, , \qquad \operatorname{div} \bm{B} = 0\, ,$$</a>
$$\dot{\bm{E}} + 2 H \bm{E} - \frac{1}{a}\operatorname{rot} \bm{B} + \bm{J} = 0 \, ,$$
$$\dot{\bm{B}}  + 2 H \bm{B} + \frac{1}{a}\operatorname{rot} \bm{E} = 0 \,$$

and use them to formulate a tower of ODEs for the quantities

$$ \mathcal{F}_\mathcal{E}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle \bm{E} \cdot \operatorname{rot}^n \bm{E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm UV}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\, ,$$
$$ \mathcal{F}_\mathcal{G}^{(n)} = -\frac{a^4}{2 k_{{\rm UV}}^{n+4}}\langle \bm{E} \cdot \operatorname{rot}^n \bm{B} + \bm{B} \cdot \operatorname{rot}^n \bm{E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)} \frac{{\rm d} k}{k} \frac{a k^{n+4}}{2 \pi^2 k_{{\rm UV}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)] \, ,$$
$$ \mathcal{F}_\mathcal{B}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle \bm{B} \cdot \operatorname{rot}^n \bm{B}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm UV}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2 \, ,$$

where, $k_{\rm UV}$ is a suitably chosen UV regulator which can vary with time. 
For completeness, we have also given the expression for $\mathcal{F}_\mathcal{X}^{(n)}$ in terms of the mode functions $A_\lambda(t,k)$.

The ODE for the $\mathcal{F}_\mathcal{X}^{(n)}$'s are then given by

$$\frac{\rm d}{{\rm d} t} \mathcal{F}_\mathcal{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_\mathcal{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_\mathcal{G}^{(n+1)} + 2 \frac{a^4}{k_{\rm UV}^{n+4}} \langle \bm{J} \cdot \operatorname{rot}^n \bm{E} \rangle =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_\mathcal{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_\mathcal{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_\mathcal{E}^{(n+1)} - \mathcal{F}_\mathcal{B}^{(n+1)}\right) - \frac{a^4}{k_{\rm UV}^{n+4}} \langle \bm{J} \cdot \operatorname{rot}^n \bm{B} \rangle= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_\mathcal{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_\mathcal{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_\mathcal{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

Although these are infinitely many coupled ODE's, one can typically determine an analytical closing condition, such that they may be truncated at some order $n_{\rm tr}$.

The ODEs for the $\mathcal{F}_\mathcal{X}^{(n)}$ can now be simply solved alongside those of the inflationary background.
This way, one can handle gauge-field backreaction onto the background dynamics of inflation.

The GEFF package is designed to help the user in the process of solving these equations in the following way
 - pre-defined and ready-to-use [GEF flavors](#basics)
 - tailored [algorithm](#algorithm) to solve the inflationary background dynamics
 - options to [implement your own GEF flavor](#model_creation)

 ---

<a name="basics">
# Sampling the GEF flavors

GEF models come in various flavors, some of the most intresting applications of the GEF are already implemented in this package.
As the first part of our tour, we explore the basic application of the GEFF code using these pre-defined models.

## Choosing your flavor

We start by sampling a first flavor of our choice; the model "pai":

```python
from geff import compile_model

# Create the GEF model of our choice
paiGEF = compile_model("pai")
``` 
The object `paiGEF` is the compiled version of the model found under `.models.pai`. It defines an ODE solver 
which needs to be initialized with information on the setup we want to study:
The "pai" model expects information on the inflaton&ndash;vector coupling, $\beta / M_{\rm P}$, 
initial conditions for the inflaton field, $\varphi(0)$, $\dot\varphi(0)$, and the shape of the inflaton potential, $V(\varphi)$.

In this example, we configure the model to start on the slow-roll attractor of a chaotic inflation potential:

$$ \varphi(0) = 15.55 M_{\rm P}, \qquad \dot{\varphi}(0) = -\sqrt{\frac{2}{3}} m M_{\rm P}, \qquad V(\varphi) = \frac{1}{2}m^2 \varphi^2$$

where $m = 6.16 \times 10^{-6} M_{\rm P}$. We set the coupling to $\beta = 15$.

The necessary information is passed to `paiGEF` as keyword arguments:
```python
import numpy as np

m = 6.16e-6
beta = 15

phi = 15.55
dphi = -np.sqrt(2/3)*m

def V(x): return 0.5*m**2*x**2
def dV(x): return m**2*x

mod = paiGEF(beta=beta, phi=phi, dphi=dphi, V=V, dV=dV)
```
If you want to know what input is expected by the GEF model, use the `print_input` method of `paiGEF`.

> **A note on units**:
> The pre-defined GEF flavors in the GEFF package work in Planck units $M_{\rm P}=1$. 
> From the input, the GEF determines the Hubble rate at initialization, $H_0$, (also in Planck units).
> Internally, the numerical routines work with dimensionless quantities, e.g., $\bar{X} = X H_0^{-a} M_{\rm P}^{-b}$ with $a$ and $b$ indicating
> how $X$ scales with an inverse timescale (e.g., $H_0$) and an energy scale (e.g., $M_{\rm P}$).
> For example, the dimensionless inflaton velocity would be like $\dot{\bar{\varphi}} = \dot{\varphi}/(H_0 M_{\rm P})$,
> i.e., $\dot{\varphi}$ scales like an amplitude, $\varphi \sim M_{\rm P}$, and a derivative, $\partial_t \sim H_0$.
> Don't worry, this is happening under the hood, but if you want more details, see `.bgtypes`.

## Getting a taste

Now, that our model is initialized, we can solve the inflationary background evolution from these initial conditions:
```python
sol, spec, info = mod.run()
```
The `run` method returned three objects. The evolution of the background dynamics is contained in `sol`,
the evolution on the gauge-field mode functions, $A_\lambda(t, k)$, are computed and returned as the
object `spec`, while `info` is just a byproduct that contains full information on the ODE solution in `sol`.
For basic applications, all information we actually want is in `sol` and `spec`.

Let us focus on `sol`. It is a `BGSystem` object which we can use to access the time evolution of several important inflationary quantities defined by our "pai" model.
For example, you can use it to make a basic plot showing the evolution of the energy densities during inflation as a function of $e$-folds

```python
import matplotlib.pyplot as plt

# Plot the evolution of the inflaton amplitude as a function of e-folds:
plt.plot(sol.N, sol.dphi**2/(6*sol.H**2)) # inflaton kinetic energy density
plt.plot(sol.N, (sol.E + sol.B)/(6*sol.H**2)) # electromagnetic energy density
plt.plot(sol.N, sol.V(sol.phi)/(3*sol.H**2)) # inflaton potential energy density

plt.yscale("log")
plt.ylim()

plt.show()
``` 

How did we know that `sol` owns the attributes `N`,`phi`, etc.? You can find out using `sol.value_names()`.
To print a full description of all available variables for "pai", you can use `paiGEF.print_ingredients()`. 
Otherwise, this information can also be found at `.models.pai`. 
If you need a brief description of any variable `X`, use `X.get_description()`.

> **A note on variables:**
> The variables encoded in a GEF model use a custom class called `Variable`. They work like a `numpy` array. 
> Additionally, constants are realized using the class `Constant`, and work like a `float`.
> Dimensionful functions like the inflaton potential use the `Func` class, and can be used like a regular function.
> These classes are defined to take care of unit conversions and are collectively attatched to a `BGSystem`. If you are curious, have a look at `.bgtypes`.

We did not only get back a `sol` object, but also `spec`. This object contains the gauge-field mode functions $A_\lambda(t,k)$.
The code computes these mode functions after having solved the dynamics of the background system ($H(t)$, $\varphi(t)$ etc.),
and uses them for estimating the convergence of the background solution. 
We briefly discuss this algorithm in [the next section](#algorithm).

The object `spec` is an instance of the `GaugeSpec` class, which defines useful methods for handling time-dependent gauge-field spectra.
For details, see `.mbm.GaugeSpec`.

> **A note on gauge-field spectra**
> If you are only interested in the background evolution and not in the gauge-field spectrum, set `nmodes=None` in `run()`.
> However, it is advised to compute gauge-field spectra to ensure consistency between the background evolution and the spectra.

Next, let us store our GEF and mode solutions in a file:
```python
# some dummy paths for illustration
gefpath = "some_gef_file.dat"
mbmpath = "some_mbm_file.dat"

sol.save_variables(gefpath)
spec.save_spec(mbmpath)
```

The data can be restored from these files using
```python
from geff import GaugeSpec

sol = mod.load_GEFdata(gefpath)
spec = GaugeSpec.read_spec(mbmpath)
```

> **A note on storage:**
> The `save_variables` method does not store information on constants or functions. So, to retrieve
> the full information on our GEF run, we need to use an appropriately configured instance of
> `paiGEF`. In the example above, this is achieved by reusing `mod`.

## A rich palette

Obtaining the inflationary dynamics is nice but it is not all the GEFF can do. For example, let's use our results to compute the corresponding gravitational-wave spectrum:
```python
from geff.tools import PowSpecT, omega_gw

# Use sol to initialize the PowSpecT class
pt_fai = PowSpecT(sol)

# Compute the vacuum and induced power spectrum for 100 momentum modes k using spec:
ks, pt_spec = pt_fai.compute_pt(100, spec)

# from the power spectrum, we can then deduce the gravitational-wave spectrum:
f, gwspec = omega_gw(ks, pt_spec["tot"], Nend=sol.N[-1], Hend=sol.H[-1])
``` 
We can just use  `sol` to extract the relevant information on the background solution.
It's that easy!

To finish this first part of our tour, let us sample a second GEF flavor, "fai_kh":
```python
# initialize the model
faiGEF = compile_model("fai_kh", {"picture":"electric"})

# chose initial conditions
mod = faiGEF(beta=...)

# solve the ODEs as before
sol, spec, info = mod.run(...)

...
...
```
This model comes in three varieties ("pictures") corresponding to the effective treatment of fermions in the model.
We specified the particular variety py passing a `settings` dictionary upon creating the model.

For more details on the available models, see `geff.models`.

<a name="algorithm">
# On the factory floor

Next on the tour of the GEF factory, we visit the production line.
The following diagram sketches the basic algorithm behind the `run` method.

```mermaid
---
config:
    flowchart:
        defaultRenderer: "elk"
    theme: 'base'
    themeVariables:
        secondaryColor: '#CFCFC6'
        tertiaryColor: '#FAFAFA'
---
graph TB
    subgraph A["`**GEF Model**`"]
        direction TB
        St((initial data <br>from model))--> GS
        subgraph GS["`**GEFSolver**`"]
            direction LR
            GS1[/initial data/] --> GS2[solve background ODEs] --> GS3[/background <br> dynamics/]
        end
        GS -.-> MbM
        subgraph MbM["`**ModeSolver**`"]
            direction LR
            MbM1[/background <br> dynamics/] --> MbM2[compute <br>mode functions] --> MbM3[/spectrum/]
        end
        MbM -.-> A2[compare <br>background dynamics <br>& spectrum]
        C[self-correction <br> using spectrum]
        GS --> A2
        A2 -.-> |disagreement| C -.->  GS
    A2 --> |agreement|Fin[finalize] --> R1((background <br> dynamics))
    Fin -.-> R2((spectrum))
    end
```

As we have seen in the [first section](#basics), the `run` method is executed as part of a GEF Model.
Each GEF Model consists of two components, the `GEFSolver` and the `ModeSolver`. The former determines the time-dependent inflationary background, 
the latter computes the mode functions $A_\lambda(t,k)$.
The two results can then be compared to eachother to assess the convergence of the background dynamics.
If the two disagree, the GEF will attempt to self-correct using $A_\lambda(t,k)$.

Note that, if you use `run(nmodes=None)`, the dotted lines in the diagram can be ignored; the background solution is immediately returned without
computing the gauge-field spectrum.

For more details on the `GEFSolver`, see `geff.solver`, while for the `ModeSolver` see `geff.mbm`.

---

<a name="model_creation">
# Create your own flavor

Having explored the potential of the GEFF code, you may be inclined to create your own GEF flavor.
To help you in this process, we show how to implement an example toy model.

> **Warning**: Before jumping into this section, we advise that you familiarize yourself with the GEF.
> Also, this tutorial works best, if you have a basic understanding of the classes defined in `.bgtypes`.

## The first step is the hardest

First, we need to work out the mathematical formulation of our model.

Let us consider the case of Abelian gauge-field production in de Sitter space ($H={\rm const.}$) by a current of the type $\bm{J} = 2 H \xi \bm{B}$, 
where $\xi$ is a constant, which we refer to as instability parameter.The ODE tower for the gauge-field bilinears are then given by:

$$\frac{\rm d}{{\rm d} t} \mathcal{F}_\mathcal{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_\mathrm{h}}{{\rm d} t} \mathcal{F}_\mathcal{E}^{(n)}  + 2\frac{k_\mathrm{h}}{a}\mathcal{F}_\mathcal{G}^{(n+1)} - 4 H \xi \mathcal{F}_\mathcal{G}^{(n)}=  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{{\rm d}}{{\rm d} t} \mathcal{F}_\mathcal{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_\mathrm{h}}{{\rm d} t} \mathcal{F}_\mathcal{G}^{(n)} - \frac{k_\mathrm{h}}{a}\left(\mathcal{F}_\mathcal{E}^{(n+1)} - \mathcal{F}_\mathcal{B}^{(n+1)}\right) - 2 H \xi \mathcal{F}_\mathcal{B}^{(n)}= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{{\rm d}}{{\rm d} t} \mathcal{F}_\mathcal{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_\mathrm{h}}{{\rm d} t} \mathcal{F}_\mathcal{B}^{(n)} - 2\frac{k_\mathrm{h}}{a}\mathcal{F}_\mathcal{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

One can determine that a sensible regularization scale for this model is given by $k_{\rm h}(t) = 2aH\xi$. 

The boundary terms $S_\mathcal{X}^{(n)}$ are a consequence of the time dependence of $k_{\rm h}$. They are expressed in terms of Whittaker functions,
but the GEFF has a module that takes care of them. We will see this later.

Once we have worked out the equations, we can turn to writing a new model file.

## Only the best ingredients

The first thing we should state in our model file is the name of the new GEF flavor:
```python
import numpy as np

name = "tutorial"
```
> **A note on settings**: Following the model's name, you can also define a `settings` dictionary.
> This dictionary should contain some key&ndash;value pair defining its name and the default setting.
> The `settings` should be accompanied by a function called `interpret_settings`. This method will be called on model creation,
> so you can use it to define how user input settings are handled. See, e.g., `.models.fai_kh` for how this is done in practice.

Next, we need to define define and categorize the variables which appear in our model. This is taken care of by the functions `define_var`, `define_const`,  and `define_func` in `.bgtypes`.
This module also contains some pre-defined variables, which are often encountered in GEF models.

There are three variables which every GEF model needs to define:
* $t$ - *cosmic time* : all ODE's are solved in terms of $t$ starting from $t=0$.
* $N$ - *$e$-folds* : needed by most internal methods. The code expects that $N(t=0)=0$.
* $H$ - *Hubble rate* : needed by most internal methods.

Beyond these staples, some extra variables appear in our ODE's:
* $\mathcal{F}_\mathcal{X}^{(n)}$ - *the gauge-field bilinears*
* $k_{\rm h}$ - *the UV regulator*
* $\xi$ -  *the instability parameter*

To properly account for the gauge field, we also need to add in the following three variables:
* $\mathcal{E}^{(0)} \equiv \langle \bm{E}^2 \rangle$ - *called `E` by the GEF*
* $\mathcal{B}^{(0)} \equiv \langle \bm{B}^2 \rangle$ - *called `B` by the GEF*
* $\mathcal{G}^{(0)} \equiv -\langle \bm{E} \cdot \bm{B} \rangle$ - *called `G` by the GEF*

> **Note on gauge field bilinears** The time evolution of the variables $\mathcal{F}_\mathcal{X}^{(n)}$ will not be saved by the GEFF code,
> since we are typically only interest in the quantities with $n=0$. The output `info` returned by the `run` method contains the full information on $\mathcal{F}_\mathcal{X}^{(n)}$,
> but only the information on $n=0$ is passed to `sol` in the form of  `E`, `B` and `G`.

All these variables need to be defined in our model file. This is done as follows:
```python
from geff.bgtypes import define_const

# We make use of the fact that a lot of these variables are pre-defined:
from geff.bgtypes import t, N, a, E, B, G, kh, GF

# We also need to define some new objects:
H = define_const("H", qu_omega=1, qu_mu=0) # Hubble rate (scales like inverse time)
xi = define_const("xi", qu_omega=0, qu_mu=0) # instability parameter
```
We use `define_const` to define the constants for our model: $H$ and $\xi$.
The Hubble rate has mass dimension one, and scales with an inverse time-scale $\omega$ as, $H = \bar{H} \omega$.
It does not scale like an amplitude $\mu$. Hence, `qu_omega=1` and `qu_mu=0`.
Similarly, $\xi$ is just a number, and we set `qu_omega=0` and `qu_mu=0`.
This information needs to be passed to properly allow for unit conversions in the code.
More information on units and scaling is given in `.bgtypes`.

All the variables we have defined serve a specific purpose in our GEF model. To inform the code of this, we need to classify each of them in one of these categories:
* **time**: a time variable (needs to be `t`)
* **dynamical**: variables whose time-evolution is determined from a differential equation.
* **gauge**: the gauge-field variable `GF` representing $\mathcal{F}_\mathcal{X}^{(n)}$.
* **static**: variables whose time-evolution is computed from other variables.
* **constant**: constants of time.
* **function**: functions of the above.

In our case, this would look as follows:
```python
quantities = {
            "time":[t], # this is mandatory!
            "dynamical":[kh], # kh is best evolved from an ODE
            "gauge":[GF], # state the obvious
            "static":[N, a, E, B, G], # directly computed from other variables
            "constant":[H, xi], # we assume de-Sitter space, and xi is a constant
            "function":[]  # our model does not need any functions
            }
```

## Write a recipe

With the variables defined, an important step towards our GEF model is already taken. 
Next, let us set up the `GEFSolver`. (For more details, see `.solver.GEFSolver`.)

Internally, the GEFF package uses `scipy.integrate.solve_ivp` to solve differential equations. 
This requires that ODE's are formulated as $\dot{\vec{y}} = f(t, \vec{y})$ with $\vec{y}$ as a `numpy` array.
However, the GEFF prefers the `BGSystem` class, which takes care of unit conversions.
So, we need to define how to translate between the two.

First up, we define how to interpret user input to initialize the array $\vec{y}$:
```python
def initial_conditions(sys, ntr):
    yini = np.zeros(1 + 3*(ntr+1))
    yini[0] = np.log(2*sys.xi*sys.H) #index 0 of yini is log(kh)

    # initialize all F_X^n as zero at indices [1:]

    return yini
``` 
Note how `sys` has attributes `xi` and `H`. These correspond to the variables we have defined in the previous step.

Importantly, upon defining `initial_conditions`, we also make a choice; which entries in $\vec{y}$ correspond to which dynamical variable.
In our simple toy model, there actually is no choice to be made: The GEFF package expects that the gauge-field variables are stored *after* all
dynamical variables, as the number of the $\mathcal{F}_\mathcal{X}^{(n)}$'s will vary depending on $n_{\rm tr}$.
Hence, $k_{\rm h}$ necessarily goes first. However, we do have the choice of evolving $\log k_{\rm h}$ instead of $k_{\rm h}$.


Next, we write the recipe used to define our GEF ODE. The ODE evolution is computed in two steps:
1. `update_values`: Update `sys` according to $\vec{y}(t)$.
2. `timestep`: Compute $\dot{\vec{y}}(t)$ from `sys`.

To define `update_values`, we can use all the variables which we have previously declared (they are assumed to be in numerical units):
```python
def update_values(t, y, sys):
    # evolution of spacetime
    sys.t.value = t
    sys.N.value = sys.t*sys.H #perfect de Sitter

    # watch out, you can use sys.X as an array,
    # but to pass to other functions, its better to use sys.X.value
    sys.a.value = np.exp(sys.N.value)

    # define how kh is computed from xi:
    sys.kh.value = np.exp(y[0]) # y[0] is log(kh)

    # use that y[1] = F_E^0, y[2] = F_B^0, y[3] = F_G^0
    rescale = (sys.kh/sys.a)**4
    sys.E.value = rescale * y[1]
    sys.B.value = rescale * y[2]
    sys.G.value = rescale * y[3]

    return
```
For `timestep`, we can make use of some pre-defined functions in the `.utility` module.

```python
from geff.utility.eom import gauge_field_ode
from geff.utility.boundary import boundary_approx
from geff.utility.general import heaviside

def timestep(t, y, sys):
    dydt = np.zeros_like(y)
    
    dlnkhdt = sys.H.value #dlnkhdt derivative
    dydt[0] = dlnkhdt

    xi = sys.xi.value

    # compute boundary terms
    W = boundary_approx(xi)

    # reshape arrays to fit gauge_field_ode
    Fcol = y[1:].shape[0]//3
    F = y[1:].reshape(Fcol,3)

    # compute the gauge-field ODEs
    dFdt = gauge_field_ode(F, sys.a, sys.kh, 2*sys.H*xi, W, dlnkhdt)
    # note that we can use 'a', 'kh' etc.;
    # 'update_values' is called before 'timestep'

    dydt[1:] = dFdt.reshape(Fcol*3) # reshape to fit dydt

    return dydt
``` 

These are all the ingredients we need to formulate the GEF ODE's. We can combine them using the `.solver.GEFSolver` class factory:

```python
from geff.solver import GEFSolver

solver = GEFSolver(initial_conditions, update_values, timestep, quantities)
``` 

> **Note**: This is just a basic `GEFSolver`. You can also define `Event` objects for a solver.
> An `Event` will check for a certain condition while the ODEs are being solved, and can terminate the solver if the condition is met.
> For example, you can define an `Event` to check if the end of inflation has been reached, or if some positive definite quantity has become negative.
> A `GEFSolver` can be configured to check for any `Event` occurrences and react to them in user-specified ways.
> For more details, see `geff.solver`.

We also should define the `ModeSolver`. In this toy model, we can just use a pre-defined class. For more complex situations, use `.mbm.ModeSolver`.
```python
from geff.mbm import BaseModeSolver

MbM = BaseModeSolver
```

> **Note on naming**: The names `solver` and `MbM` are not arbitrary. The `compile_model` function will look for objects with exactly these names.
> Please stick to the naming convention we give here to ensure your model works as intended.

## The finishing touch

 The last thing we need to do is define how our new GEF model is initialized.

First, we need to declare, what input our GEF model expects from the user.
There are two constants, $H$ and $\xi$, and the user should tell us their value:
```python
model_input = [H, xi]
``` 
Our model does not require other input; $\mathcal{F}_{\mathcal{X}}^{(n)}$ is initially set to zero, and $k_h(0)$ is determined from $\xi$ and $H$. 

The last step is to define the units of our GEF model based on the user input. This is achieved by the `define_units` function:
```python
def define_units(H):
    # The characteristic inverse time scale is the constant Hubble rate in Planck units
    freq = H
    # The charateristic energy scale is the Planck mass (in Planck units)
    amp = 1. 
    return freq, amp
```
The arguments of `define_units` necessarily needs to be a subset of `model_input`. In this case, we only need the Hubble rate, `H` to define our unit system:
The energy scale $\mu$ is the Planck mass in Planck units, while the inverse time scale is the constant Hubble rate $H$.

We are finally done! We can put everything we defined above in a file, which we call "tutorial.py", and we are good to go!

If all went well, you can now use your own GEF flavor just like the pre-defined ones:
```python
import numpy as np
from geff import compile_model

# Here, we assume you have saved your model as "tutorial.py"
import tutorial

# Pre-defined models can be initialized by strings, 
# for your own model, intialize by passing the module itself to compile_model
TutorialGEF = compile_model(tutorial)

H = 5e-6
xi = 5

mod = TutorialGEF(H=H, xi=xi)

sol, spec, info = mod.run()

...
```

# Attribution

If you use this software in your work, please cite:

```
von Eckardstein, R. (2025). GEFF: The Gradient Expansion Formalism Factory. Zenodo. https://doi.org/10.5281/zenodo.17356579

@misc{geff,
  author       = {von Eckardstein, Richard},
  title        = {GEFF: The Gradient Expansion Formalism Factory},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17356579},
  url          = {https://doi.org/10.5281/zenodo.17356579},
}

von Eckardstein, R. (2025). GEFF: The Gradient Expansion Formalism Factory - A tool for inﬂationary gauge-ﬁeld production. arXiv. https://arxiv.org/abs/2510.12644

@misc{vonEckardstein:2025jug,
    author        = {von Eckardstein, Richard},
    title         = {GEFF: The Gradient Expansion Formalism Factory - A tool for inflationary gauge-field production},
    eprint        = {2510.12644},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.CO},
    reportNumber  = {MS-TP-25-37},
    month         = oct,
    year          = 2025
}
```
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';
  mermaid.registerLayoutLoaders(elkLayouts);
</script>
"""
from .gef import compile_model
from .mbm import GaugeSpec
from .bgtypes import BGSystem

__version__ = "0.1.1"

