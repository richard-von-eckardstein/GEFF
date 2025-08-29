r"""
Welcome to our visitors tour of the **Gradient Expansion Formalism Factory**!

# What is the GEFF?

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

# GEF: A proud history since 2020

The gradient expansion formalism (or GEF) is an elegant numerical technique to determine the dynamics and backreaction of gauge-fields during inflation
by directly evolving the time-dependent quantum expectation values of the gauge-fields, 
e.g., $\langle {\bf E}^2 \rangle$,$\langle {\bf B}^2 \rangle$, $\langle {\bf E} \cdot {\bf B} \rangle$ etc.
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

for a suitably chosen regularization scale $k_{\rm UV}$. These ODE's are then given by **check these expressions**

$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)} + 2 \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf E} \rangle =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right) + \frac{a^4}{k_{\rm UV}^{n+4}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf B} \rangle= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$

which may be truncated at some order $n_{\rm tr}$ by an analytical closing condition. The source functions $S_{\mathcal{X}}^{(n)}$ account for the time variation of the UV regulator $k_{\rm UV}$.

These quantities may then be coupled to other equations to determine the backreaction of gauge-fields on the inflationary dynamics.
Once the dynamical background has been computed in this manner, it can be used to determine all manner of interesting results, e.g., the tensor power spectrum unduced by gauge fields.

# A first taste of the GEFF

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
> i.e., like a scalar amplitude $\varphi \sim M_{\rm P}$, and a time derivative $\partial_t \sim H_0$.
> Don't worry, this is happening under the hood. The GEF model takes care of everything. If you want more details, see `.bgtypes`.

Now, that our model is initialized, we can start our first GEF run:
```python
sol, spec = G.run()
```
The output `sol` contains the solution obtained by the GEF model and some useful statistics, but is not really interesting to us. All the
information about our dynamics are already stored in the object `G`:
```python
import matplotlib.pyplot as plt
# Plot the evolution of the inflaton amplitude as a function of e-folds:
plt.plot(G.N, G.phi)
plt.show()
``` 
How did we know the GEF model has the attributes `N` and `phi`? You can find out using `G.value_names()`.
If you want to know all variables the GEF knows about, use `G.quantity_names()`.

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

That's not all the GEF object `G` can do. Let's compute the tensor power spectrum:
```python
from GEFF.tools.pt import PT
#initialize the PT class using the GEF object:
P = PT(G)

# compute the vacuum and induced power spectrum for 100 momentum modes k:
ks, pt_dic = P.compute_pt(100, mbmpath)
``` 
We can just pass `G` to `PT` and all relevant information for the computation is extracted. It's that easy!

To finish out this tutorial, let us sample a second GEF flavour, "SE_kh":
```python
FermionGEF = GEF("SE_kh", {"picture":"electric"})
...
```
This model comes in three varieties ("pictures") corresponding to the effective treatment of fermions in the model.
We specified the particular variety py passing a `settings` dictionary upon creating the model.

You can get all GEF flavours at your disposal by using `GEFF.visit_the_GEFF()`.
 
# Create your own flavour

Having explored the potential of the GEFF code, you may be inclined to make your own GEF flavour.
To help you along this process, let us delve into an example model.

Let us consider the case of Abelian gauge-field production in de Sitter space by some time-dependent coupling function $\alpha$
The Lagrangian density for this model would be
$$\mathcal{L} = -\frac{1}{4} \alpha F_{\mu \nu} F^{\mu \nu}$$
such that the current in [Maxwell's equations](#max) is given by $J=\dot{f}/f {\bf E}$.
The ODE tower for the gauge-field bilinears are then neatly closed:
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} + 2 \frac{\dot{f}}{f}\right] \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + \left[(4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} +\frac{\dot{f}}{f}\right]  \mathcal{F}_{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right)= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$
One can determine that a sensible regularization scale for this model could be 
$$k_{\rm UV} \equiv k_{\rm f} = \frac{a}{2} \left[\left(\frac{\dot{f}}{f}\right)^2 - 2 \frac{\ddot{f}}{f} - 2H\frac{\dot{f}}{f} \right]$$.
and the



"""



from .gef import GEF, BaseGEF
"""from GEFF import gef, bgtypes, mbm, models, tools, solver, utility

__all__ = [gef, bgtypes, mode_by_mode, models, tools, solver, utility]"""
__version__ = "0.1"

def visit_the_GEFF():
    """**TODO**"""
    pass


