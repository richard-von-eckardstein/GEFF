val_docs = """
    A `Quantity` subclass used as basis for defining real-valued variables and constants.
     
    In addition to the basic structure defined by `Quantity`, this class can be used like an arithmetic type.
    It defines basic arithmetic operations on its instances as operations on their underlying `value` attribute.

    As a subclass of `Quantity` it inherits all its attributes and methods, ensuring 
     that `value` is changed according to `units` by using `conversion`.

    The class serves as a parent for `Variable` and `Constant`. 
    """

variable_docs = """
    A `Val` subclass representing real-valued variables evolving with cosmic time.

    Instances of this class can be used like a `Val` with `value` as a 1-D Numpy-Array.
     Indexed returns the associated index of `value`.
    """

variable_addendum = r"""
    A typical `Variable` is the scalar field velocity, $\dot\varphi = \omega\mu \dot\bar{\varphi}$    

    To define a custom `Variable` object use the class factory `define_var`.
    """


bgvar_addendum = """
    This is a subclass of `Variable` with a custom name and scaling.
    """

constant_docs = """
    A `Val` subclass representing a constant of cosmic time.

    Instances of this class can be used like a float for mathematical operations as defined by `Val`.
    """

constant_addendum = r"""
    A typical `Constant` is the inflaton--gauge-field coupling, $\beta/M_{\rm P} \sim \beta/(\bar M_{\rm P} \mu)$    

    To define a custom `Constant` object use the class factory `define_const`.
    """


bgconst_addendum = """
    This is a subclass of `Constant` with a custom name and scaling.
    """


func_docs = """
    A `Quantity` subclass representing real functions of variables like the inflaton potential.
    
    An instance of this class can be used as a function,
    evaluating the underlying method, `basefunc` depending on the state of `units`.

    In physical units, the call returns the result of `basefunc`.
    In numerical units, the call instead returns `basefunc(*args)/conversion`.  
    If called by a `Val` object, the argument is also converted according to the units of the `Val` instance
    (generically, identical to the ones of the `Func` instance).
    If instead called by a regular arithmetic data type (e.g., `float`),
      it is assumed that the argument is in the same unit system as the `Func` instance.
    """
func_addendum = r"""
    A typical object is the scalar potential, $V(\varphi) = \omega^2 \mu^2 \bar{V}(\bar{\varphi} \mu) $

    To define a custom `Func` object, use the class factory `define_func`.
    """

bgfunc_addendum = """
    This is a subclass of `Func` with a custom name and scaling.
    """

gaugefield_docs = r"""
    $\newcommand{\bm}[1]{\boldsymbol{#1}}$
    A low level class defining some basic properties of gauge-field bilinear towers.

    A gauge-field bilinear tower is defined as a collection of the following three objects,

    $$ \mathcal{F}_\mathcal{E}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle \bm{E} \cdot \operatorname{rot}^n \bm{E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm UV}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\, ,$$
    $$ \mathcal{F}_\mathcal{G}^{(n)} = -\frac{a^4}{2 k_{{\rm UV}}^{n+4}}\langle \bm{E} \cdot \operatorname{rot}^n \bm{B} + \bm{B} \cdot \operatorname{rot}^n \bm{E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)} \frac{{\rm d} k}{k} \frac{a k^{n+4}}{2 \pi^2 k_{{\rm UV}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)] \, ,$$
    $$ \mathcal{F}_\mathcal{B}^{(n)} = \frac{a^4}{k_{{\rm UV}}^{n+4}}\langle \bm{E} \cdot \operatorname{rot}^n \bm{E}\rangle = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm UV}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2 \, ,$$

    here $k_{\rm UV}$ is a UV cutoff scale, $\bm{E}$ and $\bm{B}$ are electric and magnetic field operators,
      and $A_\lambda(t,k)$ is a gauge-field mode function. The integer $n$ varies between $0$ and a maximum value, $n_{\rm tr}$.

    The `GaugeField` class collects important information about the collection $\mathcal{F}_\mathcal{X}^{(n)}$.
    The `GEFF` code needs to know, which `Variable` sets the UV cutoff scale, and
    which `Variable`s correspond to the zero-order quantities, $\langle \bm{E}^2\rangle$, $\langle \bm{B}^2\rangle$, and $-\langle \bm{E} \cdot \bm{B}\rangle$.

    Note that a `GaugeField` is never part of a `BGSystem`, as the number of variables depends on $n_{\rm tr}$, which is not fixed a priori.
    In fact, terms with $n>1$ are often only auxiliary and not used by the `GEFF` after the differential equations have been solved.
    However, the quantities $\mathcal{F}_\mathcal{X}^{(n)}$ are defined to be unitless, so they do not need to be converted between numerical and physical units.

    A custom `GaugeField` can be defined using `BGGauge`.
    """


DOCS = {
    "module":r"""
    This module defines base classes used throughout the `GEFF` module.

    The main purpose of these classes is to address the following.
    Most quantities appearing in the GEF will scale according to an inverse-time scale, $\omega$, and an energy scale, $\mu$.
    For example, a time derivative scales with inverse time, $\partial_t \sim \omega$. 
    Consequently, also a gauge field, $A_\mu \sim \omega$, as it appears in covariant derivatives.
    On the other hand, the amplitude of a scalar field scales with energy, $\varphi \sim \mu$.
    In typical inflationary contexts, $\omega \equiv H_0$, some constant Hubble rate, and $\mu \equiv M_{\rm P}$, the Planck mass.
    
    For numerical applications, it is convenient to work with dimensionless quantities.
    For example, it is useful to perform numerical computations using the dimensionless scalar amplitude $\bar{\varphi} = \varphi/\mu$, or the dimensionless gauge field, $\bar{A}_\mu = A_\mu / \omega$.
    Ultimately, the quantities we are interested in are obviously $\varphi$, $A_\mu$, etc. 
    Therefore, we need an easy way to switch between $\bar{X}$ and $X$.

    Throughout the code, we refer to the dimensionless variable, $\bar{X}$, as being in **numerical units**, while $X$ is in **physical units**. 

    To facilitate switching between these two unit systems throughout the code, this module provides the classes `BGSystem`, `Variable`, `Constant`, and `Func`. 
    The latter three are collectively referred to as `Quantity` objects.
    Each `Quantity` object is defined by a scaling with $\omega$ and $\mu$. For example, the `Quantity` $X$ scales as $X = \omega^a \mu^b \bar{X}$, where $\bar{X}$ is the re-scaled quantity used for numerical computations.
    A `BGSystem` is a collection of `Quantity` objects, which defines a common unit system by setting the value of $\omega$ and $\mu$.

    The user may define variables that evolve with cosmic time using the `define_var` class factory, which creates subclasses of `Variable`. Examples of a `Variable` are the Hubble rate, scalar field amplitude etc.
    In the same manner, the user can define constants of cosmic time using the `define_const` class factory, which creates subclasses of `Constant`. Examples of a `Constant` are, e.g., coupling strengths.
    Some quantities are functions of variables, for example, a scalar potential. These are defined by the factory `define_func`, which creates subclasses of `Func`.

    The following examples illustrates the basic use of these classes:

    Examples
    --------
    1. Defining a `BGSystem`
    ```python
    from GEFF.bgtypes import BGSystem, define_var

    # define two variables corresponding to physical time and Hubble rate.
    time = define_var("t", qu_omega=-1, qu_mu=0)
    Hubble = define_var("H", qu_omega=1, qu_mu=0)

    # Create a BGSystem with 'time' and 'Hubble'
    # We set the reference frequency to 1e-5*Mpl
    # The reference energy is the Planck mass in Planck units (so 1)
    U = BGSystem({time, Hubble}, omega=1e-5, mu=1)

    # The BGSystem knows about the new variables time and Hubble
    print(U.quantity_names()) # prints ["t", "H"]

    # However, neither the constant nor the variable are instantiated 
    # (we did not specify their value)
    print(U.variable_names()) # prints []  

    # Instantiate the quantities t and H using the units defined by U:
    U.initialise("t")(1e5)
    U.initialise("H")(1e-5)
    print(U.variable_names()) # prints ["t", "H"] 

    # The BGSystem now recognizes "t" and "H" as keys:
    print(f"In Planck units, the Hubble rate at time {U.t.value:.1e} is {U.H.value:.1e}.")
    # gives U.t=1e5, U.H=1e-5

    # Convert everything to numerical units
    U.units = False
    print(f"In numerical units, the Hubble rate at time {U.t.value} is {U.H.value:.1e}.") 
    # gives U.t=1, U.H=1

    ```

    2. Adding a new `Variable` 

    ```python
    #Let us reset the units of U:
    U.units = True

    # add a new Val: the electric-field expectation value E^2
    U.add_variable("E0", qu_omega=4, qu_mu=0) #since A_mu scales like d / dx^mu 

    # initialise E0 with some value in Planck units:
    U.initialise("E0")( 6e-10 )
    ```

    3. Operations between `Variable`'s

    ```python
    # We add a new Variable to U, the magnetic-field expectation value
    U.add_variable("B0", 4, 0)
    U.initialise("B0")(1e-10)

    # We can safely add up E0 and B0 as they are in the same units:
    print(U.E0 + U.B0) #gives 7e-10 = 6e-10 + 1e-10

    # This behaviour is not exclusive to Variables,
    # but also works for Constants:
    U.add_constant("BConst", 4, 0)
    U.initialise("BConst")(5e-11)

    print(U.E0 + U.BConst) #gives 6.5e-10 = 6e-10 + 5e-11
    ```

    4. Changing the value of a `Variable`
    ```python
    # The value of a `Variable` can be passed directly
    U.BConst.value = 1e-3
    print(str(U.BConst))

    # if we change to numerical units,
    # the argument which is passed is treated in numerical units
    U.units = False
    U.BConst.value = 1
    print(U.BConst)
    ```

    5. Adding a new `Func`

    ```python
    #first, return to physical units
    U.units = True

    # define a new Func: rhoE, the electric-field energy density
    rhoE = U.add_function("rhoE", args=[U.E0], qu_omega=2, qu_mu=2) # since 3 * M_pl^2 * H^2 = rho
    #Note how E0 is passed for creation to indicate the scaling of the argument

    # define rhoE as a function of E0:
    def func(x): return 0.5*x
    U.initialise("rhoE")( func )
    # U.rhoE is now a Callable function with a single argument
    ```

    6. Calling a `Func` with a `Val`

    ```python
    # Calling rhoE in physical units is straight forwards:
    print( U.rhoE( U.E0 ) ) # gives 3e-10 (in Planck units)

    # compare this to a direct call to func:
    print( func(6e-10) )  # gives 3e-10 (the result in Planck units)

    # Switching E0 to numerical units, nothing changes since rhoE is in physical units:
    U.E0.units = False # E0.value = 6e10
    print( U.rhoE(U.E0) ) # gives 3e-10 (still in Planck units)

    # Only switching U.rhoE to numerical units changes the result:
    U.rhoE.units = False
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.omega*U.mu)**2 (now in numerical units)

    # Again, this outcome does not depend on the units of E0:
    U.E0.units = False
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.omega*U.mu)**2 (in numerical units)
    ```

    7. Calling a `Func` with a `float`

    ```python
    # instead of calling rhoE by E0, we can call it by a float:
    val = 6e-10

    # First the behavior if rhoE is in physical units:
    U.rhoE.units = True
    print( U.rhoE( val ) ) # gives 3e-10 (in Planck units)

    # The behavior is different compared to a Val if rhoE is in numerical units:
    U.rhoE.units = False
    print( U.rhoE(val) ) #gives 3e-20 = 0.5* (6e-10*U.omega**4) / (U.omega*U.mu)**2

    # Unlike a Val, the float does not track units
    # Therefore, it is always assumed to be in the units of rhoE
    # If you want to get the correct result, you would need to convert val by hand:
    print( U.rhoE(val/U.omega**4) ) # gives 3., the expected result in numerical units.

    # Overall, its safer to just keep everything in the same units:
    U.units = True
    ```
    """,

    "BGSystem":"""
    A collection of cosmological variables sharing a system of units.

    Instances of this class define two base unit systems,
    *physical units* and *numerical units*, by setting an energy scale, `mu`, and an inverse time scale, `omega`. 

    The cosmological variables (time, Hubble rate, etc.) are represented by `Quantity` objects.
    These objects are stored in `quantities`, and can can be initialise using `initialise`.
    Instances of these objects can be collectively converted between units by setting `units`. 

    This class is the fundamental building block of the `GEFF` code. 
    """,

    "Quantity":r"""
    An object representing a cosmological quantity. 

    A cosmological quantity has a characteristic scaling with respect to a frequency scale (e.g., the Hubble rate at some reference time), and energy scale (e.g., the Planck mass).

    Typical such objects are:
    - cosmic time $t = \bar{t}/\omega$
    - frequency $f = \omega \bar{f}$
    - scalar-field vev $\varphi = \mu \bar{\varphi}$
    - scalar potential $ V(\varphi) = \omega^2 \mu^2 \bar{V}(\bar{\varphi} \mu) $ 
    - gauge fields $A_\mu =  \omega\bar{A}_\mu $ *(i.e., scales like a time derivative)*

    `Quantity` objects are initialized as part of a `BGSystem`, which defines $\omega$ and $\mu$.

    This class is a parent of `Val` and `Func`.
    """,

    "Val":val_docs,
    "Variable":variable_docs,
    "Constant":constant_docs,
    
    "define_var.CustomVar":variable_docs+bgvar_addendum,
    "define_const.CustomConst":variable_docs+bgvar_addendum,

    "Func":func_docs+ func_addendum,
    "define_func.CustomFunc":val_docs+bgfunc_addendum,

    "GaugeField":gaugefield_docs,
}

