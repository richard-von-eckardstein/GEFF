r"""
Welcome to the **Gradient Expansion Formalism Factory**!

# What is the GEFF?

Use this python package to investigate gauge-field production during cosmic inflation.

If you are interested in axion inflation, the package comes with everything you need:
- **pre defined models**
    - pure axion inflation (PAI) : *good old axion inflation*
    - fermionic axion inflation (FAI) : *coupled to Standard Model fermions!*
- **useful tools**
    - Resolve the dynamics of axion inflation including homogeneous backreaction.
    - Investigate the gauge-field spectrum.
    - Determine the vacuum and induced tensor power spectrum.
    - Compute gravitational wave spectra.


But we don't want to hold you back! The package provides a flexible framework to define your **own GEF model**, with all tools at your disposable. 
It is indeed a true GEF *factory*!

# The gradient expansion formalism

The gradient expansion formalism (or *GEF*) is an elegant numerical technique to determine the dynamics and backreaction of gauge-fields during inflation
by directly evolving the time-dependent quantum expectation values of the gauge-fields, 
e.g., $\langle {\bf E}^2 \rangle$,$\langle {\bf B}^2 \rangle$, $\langle {\bf E} \cdot {\bf B} \rangle$ etc.
If this is the first time you encounter the GEF, here are some useful articles on the topic:
* ...
* ...

Summarized, the strategy is to take the (charge-free) Maxwell's equations in an expanding spacetime,
$$\operatorname{div} {\bf E} = 0\, , \qquad \operatorname{div} {\bf B} = 0\, ,$$
$$\dot{{\bf E}} + 2 H {\bf E} - \frac{1}{a}\operatorname{rot} {\bf B}  = {\bf J} \, ,$$
$$\dot{{\bf B}}  + 2 H {\bf B} + \frac{1}{a}\operatorname{rot} {\bf E} = 0 \,$$
and reformulate them into an infinite tower of ODE's
**check these expressions**
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{E}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{E}^{(n)}  + 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)} - \frac{a^4}{k_{\rm UV}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf E} \rangle =  S_{\mathcal{E}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{G}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{G}^{(n)} - \frac{k_{\rm UV}}{a}\left(\mathcal{F}_{E}^{(n+1)} - \mathcal{F}_{B}^{(n+1)}\right) -\frac{a^4}{2k_{\rm UV}} \langle {\bf J} \cdot \operatorname{rot}^n {\bf B} \rangle= S_{\mathcal{G}}^{(n)}\, , $$
$$\frac{\rm d}{{\rm d} t} \mathcal{F}_{B}^{(n)} + (4+n)\frac{{\rm d} \ln k_{\rm UV}}{{\rm d} t} \mathcal{F}_{B}^{(n)} - 2\frac{k_{\rm UV}}{a}\mathcal{F}_{G}^{(n+1)}  =  S_{\mathcal{B}}^{(n)}\, .$$
which you can truncate at some order $n_{\rm tr}$ by an analytical closing condition.

The variables, $\mathcal{F}_\mathcal{X}^{(n)}$ encodes the variables
$$ \mathcal{F}_{E}^{(n)} = \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf E}\rangle\, ,$$
$$ \mathcal{F}_{G}^{(n)} = -\frac{a^4}{2 k_{\mathrm{h}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf B} + {\bf B} \cdot \operatorname{rot}^n {\bf E}\rangle\, ,$$
$$ \mathcal{F}_{B}^{(n)} = \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle {\bf E} \cdot \operatorname{rot}^n {\bf E}\rangle$$
for a suitably chosen regularisation scale $k_{\rm UV}$.

These quantities may then be coupled to other equations to determine the backreaction of gauge-fields on the inflationary dynamics.

# First steps
"""



from .gef import GEF, BaseGEF
"""from GEFF import gef, bgtypes, mbm, models, tools, solver, utility

__all__ = [gef, bgtypes, mode_by_mode, models, tools, solver, utility]"""
__version__ = "0.1"



