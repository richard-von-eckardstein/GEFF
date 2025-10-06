r"""
Welcome to our visitors tour of the **Gradient Expansion Formalism Factory**!

# GEFF: Established in 2025

Use this python package to investigate gauge-field production during cosmic inflation.

If you are interested in axion inflation, the package comes with everything you need:
- **a variety of flavors**
    - pure axion inflation (PAI) : *good ol' axion inflation*
    - fermionic axion inflation (FAI) : *axion inflation with Standard Model fermions!*
- **useful tools**
    - Resolve the dynamics of axion inflation including homogeneous backreaction.
    - Analyze the gauge-field spectrum.
    - Determine the vacuum and induced tensor power spectrum.
    - Compute gravitational-wave spectra.


But we don't want to hold you back! The package provides a flexible framework to create your **own GEF flavor**, with all tools at your disposable. 
It is indeed a true GEF *factory* !

# The refreshing taste of GEF

The *gradient expansion formalism* (or GEF) is a numerical technique to determine the dynamics and backreaction of gauge-fields during inflation
by directly evolving the time-dependent quantum expectation values of the gauge field, 
e.g., $\langle {\bf E}^2 \rangle$, $\langle {\bf B}^2 \rangle$, $\langle {\bf E} \cdot {\bf B} \rangle$ etc.
If this is the first time you encounter the GEF, here are some useful articles on the topic:
* ...
* ...

The strategy behind the GEF is to take Maxwell's equations in an expanding spacetime,

<a name="max">$$\operatorname{div} {\bf E} = 0\, , \qquad \operatorname{div} {\bf B} = 0\, ,$$</a>
$$\dot{{\bf E}} + 2 H {\bf E} - \frac{1}{a}\operatorname{rot} {\bf B} + {\bf J} = 0 \, ,$$
$$\dot{{\bf B}}  + 2 H {\bf B} + \frac{1}{a}\operatorname{rot} {\bf E} = 0 \,$$

and use them to formulate a tower of ODEs for the quantities

$$ \mathcal{F}_{E}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm UV}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\, ,$$
$$ \mathcal{F}_{G}^{(n)} = -\frac{a^4}{2 k_{{\rm UV}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf B} + {\bf B} \cdot \operatorname{rot}^n {\bf E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)} \frac{{\rm d} k}{k} \frac{a k^{n+4}}{2 \pi^2 k_{{\rm UV}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)] \, ,$$
$$ \mathcal{F}_{B}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle {\bf B} \cdot \operatorname{rot}^n {\bf B}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm UV}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2 \, ,$$

where, $k_{\rm UV}$ is a suitably chosen UV regulator which can vary with time. 
For completeness, we have also given the expression for $\mathcal{F}_{X}^{(n)}$ in terms of the mode functions $A_\lambda(t,k)$.

The ODE for the $\mathcal{F}_{X}^{(n)}$'s are then given by

$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)} + 2 \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf E} \rangle =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right) + \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf B} \rangle= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

Although these are infinitely many ODE's, one can typically determine an analytical closing condition, such that they may be truncated at some order $n_{\rm tr}$.

These ODEs for $\mathcal{F}_{X}^{(n)}$ can now be simply solved alongside those of the inflationary background.
This way, one can handle gauge-field backreaction onto the background dynamics of inflation.

The GEFF package is designed to help the user in the process of solving these equations in the following way
 - pre-defined and ready-to-use [GEF flavors](#basics)
 - tailored [algorithm](#algorithm) to solve the background dynamics
 - options to [implement your own GEF flavor](#model_creation)

<a name="basics">
# Sampling the GEF flavors

GEF models come in various flavors, some of the most intresting applications of the GEF are already implemented in this package.
As the first part of our tour, we explore the basic usage of the GEFF code.

## Choosing your flavor

We start by sampling a first flavor of our choice; the model "classic", which corresponds to PAI:

```python
from GEFF import GEF

# Create the GEF model of your choice
ClassicGEF = GEFModel("classic")
``` 
To use `ClassicGEF`, we need to initialize it by passing initial conditions for our background evolution. 
The model expects information on the inflaton--vector coupling $\beta / M_{\rm P}$, 
initial conditions for the inflaton field, $\varphi(0)$, $\dot\varphi(0)$, and the shape of the inflaton potential, $V(\varphi)$.

In this example, we configure the model to start on the slow-roll attractor of a chaotic inflation potential:

$$ \varphi(0) = 15.55 M_{\rm P}, \qquad \dot{\varphi}(0) = -\sqrt{\frac{2}{3}} m M_{\rm P}, \qquad V(\varphi) = \frac{1}{2}m^2 \varphi^2$$

where we set $m = 6.16 \times 10^{-6} M_{\rm P}$. We set the coupling to $\beta = 15$.

The necessary information is passed to `ClassicGEF` as keyword arguments:
```python
import numpy as np

m = 6.16e-6
beta = 15

phi = 15.55
dphi = -np.sqrt(2/3)*m

def V(x): return 0.5*m**2*x**2
def dV(x): return m**2*x

G = ClassicGEF(beta=beta, phi=phi, dphi=dphi, V=V, dV=dV)
```
If you want to know what input is expected by the GEF model, use the `print_kwargs` method of `ClassicGEF`.

> **A note on units**:
> The pre-defined GEF flavors in the GEFF package work in Planck units $M_{\rm P}=1$. 
> From the input, the GEF determines the Hubble rate at initialization, $H_0$, (also in Planck units).
> Internally, the numerical routines work with dimensionless quantities, e.g., $\bar{X} = X H_0^{-a} M_{\rm P}^{-b}$ with $a$ and $b$ indicating
> how $X$ scales with an inverse timescale (e.g., $H_0$) and an energy scale (e.g., $M_{\rm P}$).
> For example, the inflaton velocity scales like $\dot{\varphi} = \dot{\bar{\varphi}}H_0 M_{\rm P}$,
> i.e., like an amplitude, $\varphi \sim M_{\rm P}$, and a derivative, $\partial_t \sim H_0$.
> Don't worry, this is happening under the hood, but if you want more details, see `.bgtypes`.

## Getting a taste

Now, that our model is initialized, we can start determine the inflationary background evolution starting from the initial conditions:
```python
sol, spec = G.run()
```
The output `sol` contains the solution obtained by the GEF model and some useful statistics, but is only a byproduct. All the
information about our dynamical system have already been moved to the object `G`. 

For example, you can make a basic plot showing the evolution of the energy densities during inflation as a function of $e$-folds
```python
import matplotlib.pyplot as plt
# Plot the evolution of the inflaton amplitude as a function of e-folds:
plt.plot(G.N, G.phi**2/(6*G.H**2))
plt.plot(G.N, (G.E + G.B)/(6*G.H**2))
plt.plot(G.N, G.V(G.phi)/(3*G.H**2))

plt.yscale("log")
plt.ylim()

plt.show()
``` 
How did we know the GEF model has the attributes `N`,`phi`, etc.? You can find out using `G.value_names()`.
To print a full description of all available variables, use `G.print_known_quantities()`. 
If you need a brief description of the variable `X`, use `G.X.get_description()`.

> **A note on variables:**
> The variables encoded in a GEF model use a custom class called `Variable`. They work like a `numpy` array. 
> Additionally, constants are realized using the class `Constant`, and work like a `float`.
> Dimensionful functions like the inflaton potential use the `Func` class, and can be used like a regular function.
> These classes are defined to take care of unit conversions. If you are curious, have a look at `.bgtypes`.

We did not only get back a `sol` object, but also `spec`. This object contains the gauge-field mode functions $A_\lambda(t,k)$.
The code computes these mode functions after having solved the dynamics of the background system ($H(t)$, $\varphi(t)$ etc.),
and uses them for estimating the convergence of the background solution. 
We briefly discuss this algorithm in [the next section](#algorithm).

The object `spec` is an instance of the `GaugeSpec` class, which defines useful methods for handling time-dependent gauge-field spectra.
For details, see `mbm.GaugeSpec`.

> **A note on gauge-field spectra**
> If you are only interested in the background evolution and not in the gauge-field spectrum, set `nmodes=None` in `run()`.
> However, it is advised to compute gauge-field spectra to ensure consistency between the background evolution and the spectra.

Next, let us store our GEF and mode solutions in a file:
```python
# some dummy paths for illustration
gefpath = "some_gef_file.dat"
mbmpath = "some_mbm_file.dat" 
G.save_GEFdata(gefpath)
spec.save_spec(mbmpath)
```

> **A note on storage:**
> All quantities are stored as dimensionless variables, $\bar{X}$, by the GEF code.
> We therefore recommend you use the GEF class to also load the data, `G.load_GEFdata()`.
> This ensures you are using the same reference scale to restore dimensions.

## A rich palette

Obtaining the inflationary dynamics is nice, but it is not all the GEFF can do. For example, let's use our results to compute the corresponding gravitational-wave spectrum:
```python
from GEFF.tools import PT, omega_gw

# Use the GEF's knowledge of the background expansion
# to compute the vacuum and induced power spectrum for 100 momentum modes k:
ks, pt_dic =  PT(G).compute_pt(100, mbmpath)

# from the power spectrum, we can then compute the gravitational-wave spectrum:
f, h2omega_gw = omega_gw(ks, pt_dic["tot"], Nend=G.N[-1], Hend=G.H[-1])
``` 
We can just use  `G` to extract the relevant information on the background solution.
It's that easy!

To finish this first part of our tour, let us sample a second GEF flavor, "SE_kh":
```python
FermionGEF = GEFModel("SE_kh", settings={"picture":"electric"})
...
```
This model comes in three varieties ("pictures") corresponding to the effective treatment of fermions in the model.
We specified the particular variety py passing a `settings` dictionary upon creating the model.

For more details on the available models, see `GEFF.models`.

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
    subgraph A["`**GEFModel**`"]
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

As we have seen in the [first section](#basics), the `run` method is executed as part of a `GEFModel`.
Each `GEFModel` consists of two components, the `GEFSolver` and the `ModeSolver`. The former determines the time-dependent inflationary background, 
the latter computes the mode functions $A_\lambda(t,k)$.
The two results can then be compared to eachother, to assess the convergence of the background dynamics.
If the two disagree, the GEF will attempt to self-correct using $A_\lambda(t,k)$.

Note that, if you use `run(nmodes=None)`, the dotted lines in the diagram can be ignored; the background solution is immediately returned without
computing the gauge-field spectrum.

For more details on the `GEFSolver`, see `GEFF.solver`, while for the `ModeSolver` see `GEFF.mbm`.

<a name="model_creation">
# Create your own flavor

Having explored the potential of the GEFF code, you may be inclined to create your own GEF flavor.
To help you in this process, we show how to implement an example toy model.

## The first step is the hardest

First, we need to work out the mathematical formulation of our model.

Let us consider the case of Abelian gauge-field production in de Sitter space by a current of the type ${\bf J} = \xi(t) {\bf B}$.
The ODE tower for the gauge-field bilinears are then given by:
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} + 2 \frac{\dot{f}}{f}\right] \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm h}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} +\frac{\dot{f}}{f}\right]  \mathcal{F}_{G}^{(n)} - \frac{k_{\rm h}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right)= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm h}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm h}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

One can determine that a sensible regularization scale for this model is given by $k_{\rm h}(t) = 2aH \underset{s \leq t}{\max}(|\xi(s)|)$. 

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
> so you can use it to define how user input settings are handled. See, e.g., `.models.SE_kh` for how this is done in practice.

Next, we need to define define and categorize the variables which appear in our model. This is taken care of by the functions `.bgtypes.BGVar`, `.bgtypes.BGConst`,  and `.bgtypes.BGFunc`.
The `bgtypes` module also contains some pre-defined variables, which are often encountered in GEF models.

There are three variables which every GEF model needs to define:
* $t$ - *cosmic time* : all ODE's are solved in terms of $t$ starting from $t=0$.
* $N$ - *$e$-folds* : needed by most internal methods. It should be defined such that $N(t=0)=0$.
* $H$ - *Hubble rate* : needed by most internal methods.

Beyond these staples, some extra variables appear in our ODE's:
* $\mathcal{F}_{X}^{(n)}$ - *the gauge-field bilinears*
* $k_{\rm h}$ - *the UV regulator*
* $\xi(t)$ -  *the instability parameter as a function of time*
* $\dot{\xi}(t)$ - *the time derivative of $\xi$ (needed to compute ${\rm d} \ln k_{\rm h} / {\rm d}t$)*

To properly account for the gauge-fields, we also need to add in the following three variables:
* $\mathcal{E}^{(0)} = \langle {\bf E}^2 \rangle$ - *called `E` by the GEF*
* $\mathcal{B}^{(0)} = \langle {\bf B}^2 \rangle$ - *called `B` by the GEF*
* $\mathcal{G}^{(0)} = -\langle {\bf E} \cdot {\bf B} \rangle$ - *called `G` by the GEF*

> **Note on gauge field bilinears** The time evolution of the variables $\mathcal{F}_{X}^{(n)}$ will not be saved by the GEFF code,
> since we are typically only interest in the quantities with $n=0$. The output `sol` returned by the `run` method contains the full information on the $\mathcal{F}_{X}^{(n)}$,
> but only the information on $n=0$ is stored in the GEF object in the form of  `E`, `B` and `G`.

All these variables and function nned to be defined in our model file. This is done as follows:
```python
from GEFF.bgtypes import BGConst, BGFunc

# We make use of the fact that a lot of these variables are pre-defined:
from GEFF.bgtypes import t, N, a, E, B, G, kh, GF

# We also need to define some new objects:
H = BGConst("H", qu_omega=1, qu_mu=0) # Hubble rate (scales like inverse time)
xi = BGFunc("xi", [t], qu_omega=0, qu_mu=0) # dimensionless function of t
dxi = BGFunc("xi", [t], qu_omega=0, qu_mu=0) # dimensionless function of t
```
We use `BGConst` to define a new constant for our model, the Hubble rate $H$.
The Hubble rate has mass dimension one, and scales with an inverse time-scale $\omega$ as, $H = \bar{H} \omega$.
It does not scale like an amplitude $\mu$. Hence, `qu_omega=1` and `qu_mu=0`.
This information needs to be passed to properly allow for unit conversions in the code.
In the same way, we define $\xi$ and $\dot{\xi}$ as dimensionless functions of time by using `BGFunc`.
More information on units and scaling is given in `.bgtypes`.

All the variables we have defined serve a specific purpose in our GEF model. To inform the GEFF of this, we need to classify each of them in one of these categories:
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
            "constant":[H], # we assume de-Sitter space
            "function":[xi, dxi]  # a priori undetermined functions of t
            }
```

## Write a recipe

With the variables defined, an important step towards our GEF model is already taken. Next, let us set up the differential equation solver.

Internally, the GEFF package uses `scipy.integrate.solve_ivp` to solve differential equations. 
This requires that ODE's are formulated as $\dot{\vec{y}} = f(t, \vec{y})$ with $\vec{y}$ as a `numpy` array.
However, the GEFF prefers the `.bgtypes.BGSystem` class, which takes care of unit conversions.
So, we need to define how to translate between the two.

First up, we define how to interpret user input to initialize the array $\vec{y}$:
```python
def initial_conditions(sys, ntr):
    yini = np.zeros(1 + 3*(ntr+1))
    yini[0] = np.log(2*sys.xi(0)*sys.H) #index 0 is log(kh)
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

To define `update_values`, we can use all the variables which we have previously declared:
```python
# Note how we can use the Val objects like arrays, and the Func objects like functions
def update_values(t, y, sys):
    # evolution of spacetime
    sys.t.value = t
    sys.N.value = sys.t*sys.H #perfect de Sitter
    sys.a.value = np.exp(sys.N)

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
from GEFF.utility.eom import gauge_field_ode
from GEFF.utility.boundary import boundary_approx
from GEFF.utility.general import heaviside

def timestep(t, y, sys):
    dydt = np.zeros_like(y)

    xi_t = sys.xi(sys.t)

    # achieving a monotonic growth for dlnkhdt is a bit tricky:
    dlnkhdt = sys.dxi(sys.t)/abs(xi_t) #first guess for derivative
    logfc = sys.N + np.log( 2*abs(sys.xi(t))*sys.H) #non-monotonic kh
    # ensure monotonicity using heaviside functions
    dlnkhdt *= heaviside(dlnkhdt, 0)*heaviside(logfc, y[3]*(1-1e-5))
    dydt[0] = dlnkhdt

    # compute boundary terms
    W = boundary_approx(xi_t)

    # reshape arrays to fit gauge_field_ode
    Fcol = y[1:].shape[0]//3
    F = y[1:].reshape(Fcol,3)
    
    # compute the gauge-field ODEs
    dFdt = gauge_field_ode(F, sys.a, sys.kh, 2*sys.H*sys.xi_t, W, dlnkhdt)
    # note that we can use 'a', 'kh' etc.;'update_values' is called before 'timestep'
    # reshape to fit dydt
    dydt[1:] = dFdt.reshape(Fcol*3)

    return dydt
``` 

These are all the ingredients we need to formulate the GEF ODE's. We can combine them using the `.solver.GEFSolver` class factory:

```python
# pass everything we have defined to GEFSolver.
solver = GEFSolver(initial_conditions, update_values, compute_timestep, quantities)
``` 

We also should define the `ModeSolver`. In this toy model, we can just use a pre-defined class. For more complex situations, use `.mbm.ModeSolver`.
```python
from GEFF.mbm import BaseModeSolver

MbM = BaseModeSolver
```

## The finishing touch

 The last thing we need to do is define how our new GEF model is initialized.

First, we need to declare, what input our GEF model expects from the user.
The user should definately tell us the value of the Hubble rate.
Also, the instability function $\xi(t)$, and its derivative, $\dot{\xi}$, need to be passed on initialization,
similarly to how the "classic" model needed $V$ and $V_{,\varphi}$ as input:
```python
model_input = [H, xi, dxi]
``` 
Our model does not require other input; $\mathcal{F}_{\mathcal{X}}^{(n)}$ is initially set to zero, and $k_h$ is determined from $t$ and $\xi$. 

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

We are finally done! We can put everything we defined above in a file, let's call it "tutorial.py", and we are good to go!

If all went well, you can now use your own GEF flavor just like the pre-defined ones:
```python
import numpy as np
from GEFF import GEF
import tutorial

TutorialGEF = GEFModel(tutorial)

H = 5e-6
def xi(x): return 5*(np.sin(np.pi*x/5)+1) 
def dxi(x): return np.pi*np.cos(np.pi*x/5)

G = TutorialGEF(H=H, xi=xi, dxi=dxi)

G.run()
...
```
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';
  mermaid.registerLayoutLoaders(elkLayouts);
</script>
"""
from .gef import GEF, BaseGEF

__version__ = "0.1"

