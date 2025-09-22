val_docs = r"""
    A `Quantity` subclass used as basis for defining real-valued variables and constants.
     
    In addition to the basic structure defined by `Quantity`, this class can be used like an arithmetic type.
    It defines basic arithmetic operations on its instances as operations on their underlying `value` attribute.

    As a subclass of `Quantity` it inherits all its attributes and methods, but re-defines the `set_units` method
    such that changing units converts `value` according to `Quantity.get_conversion`.

    The class serves as a parent for `Variable` and `Constant`. 
    """

variable_docs = r"""
    A `Val` subclass representing real-valued variables evolving with cosmic time.

    Instances of this class can be used like a 1-D Numpy-Array for mathematical operations as defined by `Val`.
     Indexed returns the associated index of `value`.
    """

variable_addendum = r"""
    A typical `Variable` is the scalar field velocity, $\dot\varphi = \omega\mu \dot\bar{\varphi}$    

    To define a custom `Variable` object use the class factory `BGVar`.
    """


bgvar_addendum = """
    This is a subclass of `Variable` with a custom name and scaling.
    """

constant_docs = r"""
    A `Val` subclass representing a constant of cosmic time.

    Instances of this class can be used like a float for mathematical operations as defined by `Val`.
    """

constant_addendum = r"""
    A typical `Constant` is the inflaton--gauge-field coupling, $\beta/M_{\rm P} \sim \beta/(\bar M_{\rm P} \mu)$    

    To define a custom `Constant` object use the class factory `BGConst`.
    """


bgconst_addendum = """
    This is a subclass of `Constant` with a custom name and scaling.
    """


func_docs = r"""
    A `Quantity` subclass representing real functions of variables like the inflaton potential.
    
    An instance of this class can be used as a function,
    evaluating the underlying method, `f(*args)` depending on the current units.

    In physical units, the call returns the result of the underlying function, `f(*args)`.
    In numerical units, the call instead returns `f(*args)/conversion_factor`, 
    with `conversion_factor` defined by `Quantity.get_conversion`.  
    If called by a `Val` object, the argument is also converted according to the units of the `Val` instance
    (generically, identical to the ones of the `Func` instance).
    If instead called by a regular arithmetic data type (e.g., `float`),
      it is assumed that the argument is in the same unit system as the `Func` instance.
    """
func_addendum = r"""
    A typical object is the scalar potential, $V(\varphi) = \omega^2 \mu^2 \bar{V}(\bar{\varphi} \mu) $

    To define a custom `Func` object, use the class factory `BGFunc`.
    """

bgfunc_addendum = """
    This is a subclass of `Func` with a custom name and scaling.
    """


DOCS = {
    "module":r"""
    This module defines base classes used throughout the `GEFF` module.

    The main purpose of this module is to address the following situation:
    Most cosmological quantities will scale according to an inverse-time scale, $\omega$, and an energy scale, $\mu$.
    For example, a time derivative scales with inverse time, $\partial_t \sim \omega$, and therefore also a gauge field, $A_\mu \sim \omega$, as it appears in covariant derivatives.
    On the other hand, the amplitude of a scalar field scales with energy, $\varphi \sim \mu$.
    In typical inflationary contexts, $\omega = H_0$, some constant Hubble rate, and $\mu = M_{\rm P}$, the Planck mass.
    
    For numerical applications, it is convenient to work with dimensionless quantities.
    For example, it is useful to perform numerical computations using the *dimensionless* scalar amplitude $\bar{\varphi} = \varphi/\mu$, or the *dimensionless* gauge field, $\bar{A}_\mu = A_\mu / \omega$.
    Ultimately, the quantities we are interested in are obviously $\varphi$, $A_\mu$, etc. 
    Therefore, we need an easy way to switch between $\bar{X}$ and $X$. This is the purpose of this module.

    Throughout the code, we refer to the dimensionless variable, $\bar{X}$, as being in *numerical units*, while $X$ is in *physical units*. 

    To facilitate switching between these two unit systems throughout the code, this module provides the classes `BGSystem`, `Variable`, `Constant`, and `Func`. 
    The latter three are collectively referred to as `Quantity` objects.
    Each `Quantity` object is defined by a scaling with $\omega$ and $\mu$. For example, the `Quantity` $X$ scales as $X = \omega^a \mu^b \bar{X}$, where $\bar{X}$ is the re-scaled quantity used for numerical computations.
    A `BGSystem` is a collection of `Quantity` objects, which defines a common unit system by setting the value of $\omega$ and $\mu$.

    The user may define variables that evolve with cosmic time using the `BGVar` class factory, which creates subclasses of `Variable`. Examples of a `Variable` are the Hubble rate, scalar field amplitude etc.
    In the same manner, the user can define constants of cosmic time using the `BGConst` class factory, which creates subclasses of `Constant`. Examples of a `constant` are, e.g., coupling strengths.
    Some quantities are functions of variables, for example, a scalar potential. These are defined by the factory `BGFunc`, which creates subclasses of `Func`.

    The following examples illustrates the basic use of these classes:

    Examples
    --------
    1. Defining a `BGSystem`
    ```python
    from GEFF.bgtypes import BGSystem, BGVal

    # define two variables corresponding to physical time and Hubble rate.
    time = BGVar("t", q_u_omega=-1, q_u_mu=0)
    Hubble = BGVar("H", q_u_omega=1, q_u_mu=0)

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
    print(f"In Planck units, the Hubble rate at time {U.t} is {U.H}.")
    # gives U.t=1e5, U.H=1e-5

    # Convert everything to numerical units
    U.set_units(False)
    print(f"In numerical units, the Hubble rate at time {U.t} is {U.H}.") 
    # gives U.t=1, U.H=1
    ```

    2. Adding a new `Variable` 

    ```python
    #Let us reset the units of U:
    U.set_units(True)

    # add a new Val: the electric-field expectation value E^2
    U.add_variable("E0", q_u_omega=4, q_u_mu=0) #since A_mu scales like d / dx^mu 

    # initialise E0 with some value in Planck units:
    U.initialise("E0")( 6e-10 )
    ```

    3. Operations between `Variable`'s

    ```python
    # We add a new BGVal to U, the magnetic-field expectation value
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

    4. Adding a new `Func`

    ```python

    # define a new Func: rhoE, the electric-field energy density
    rhoE = U.add_function("rhoE", func_args=[U.E0], q_u_omega=2, q_u_mu=2) # since 3 * M_pl^2 * H^2 = rho
    #Note how E0 is passed for creation to indicate the scaling of the argument

    # define rhoE as a function of E0:
    def func(x): return 0.5*x
    U.initialise("rhoE")( func )
    # U.rhoE is now a Callable function with a single argument
    ```

    2. Calling a `Func` with a `Val`

    ```python
    # Calling rhoE in physical units is straight forwards:
    print( U.rhoE( U.E0 ) ) # gives 3e-10 (in Planck units)

    # compare this to a direct call to func:
    print( func(6e-10) )  # gives 3e-10 (the result in Planck units)

    # Switching E0 to numerical units, nothing changes since rhoE is in physical units:
    U.E0.set_units(False) # E0.value = 6e10
    print( U.rhoE(U.E0) ) # gives 3e-10 (still in Planck units)

    # Only switching U.rhoE to numerical units changes the result:
    U.rhoE.set_units(False)
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.omega*U.mu)**2 (now in numerical units)

    # Again, this outcome does not depend on the units of E0:
    U.E0.set_units(True)
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.omega*U.mu)**2 (in numerical units)
    ```

    3: Calling a `Func` with a `float`

    ```python
    # instead of calling rhoE by E0, we can call it by a float:
    val = 6e-10

    # First the behavior if rhoE is in physical units:
    U.rhoE.set_units(True)
    print( U.rhoE( val ) ) # gives 3e-10 (in Planck units)

    # The behavior is different compared to a Val if rhoE is in numerical units:
    U.rhoE.set_units(False)
    print( U.rhoE(val) ) #gives 3e-20 = 0.5* (6e-10*U.omega**4) / (U.omega*U.mu)**2

    # Unlike a Val, the float does not track units
    # Therefore, it is always assumed to be in the units of rhoE
    # If you want to get the correct result, you would need to convert val by hand:
    print( U.rhoE(val/U.omega**4) ) # gives 3., the expected result in numerical units.

    # Overall, its safer to just keep everything in the same units:
    U.set_units(True)
    ```
    """,

    "BGSystem":"""
    A collection of cosmological variables sharing a system of units.

    Instances of this class define two base unit systems,
    *physical units* and *numerical units*, by setting an energy scale, `omega` and an inverse time scale, `mu`. 

    The cosmological variables (time, Hubble rate, etc.) are represented by `Quantity` objects.
    These objects are stored in `quantities`, and can can be initialise using `initialise`.
    Instances of these objects can be collectively converted between units by using the scales `omega` and `mu`. 

    This class is the fundamental building block of the `geff` code. 
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
    
    "BGVar.CustomVar":variable_docs+bgvar_addendum,
    "BGConst.CustomConst":variable_docs+bgvar_addendum,

    "Func":func_docs+ func_addendum,
    "BGFunc.CustomFunc":val_docs+bgfunc_addendum

}

