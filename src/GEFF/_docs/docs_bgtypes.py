val_docs = r"""
    A `Quantity` subclass representing real-valued variables like cosmic time or Hubble rate.
     
    Instances of this class can be used like a 1-D Numpy-Array for mathematical operations and indexing.
    This class defines basic arithmetic operations for its instances as operations on their underlying 1-D Numpy-Array, `value`.  

    As a subclass of `Quantity` it inherits all its attributes and methods, but re-defines the `set_units` method
    such that changing units converts `value` according to `Quantity.get_conversion`.
    """

val_addendum = r"""
    A typical object is the scalar field velocity, $\dot\varphi \sim HM$    

    To define a custom `Val` object use the class factory `BGVal`.
    """


bgval_addendum = """
    This is a subclass of `Val` with a custom name and scaling.
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
    A typical object is the scalar potential, $V(\varphi) \sim (H M)^2$

    To define a custom `Func` object, use the class factory `BGFunc`.
    """

bgfunc_addendum = """
    This is a subclass of `Func` with a custom name and scaling.
    """


DOCS = {
    "module":r"""
    This module defines base classes used throughout the `GEFF` module.

    The main purpose of this module is to address the following situation:
    Most cosmological variables will scale according to an inverse-time scale, $H$, and an energy scale, $M$.
    In typical inflationary contexts, these scales are some constant Hubble rate, and the Planck mass.
    For example, a time derivative scales with inverse time, $\partial_t \sim H$, and therefore also a gauge field, $A_\mu \sim H_0$, as it appears in covariant derivatives.
    On the other hand, the amplitude of a scalar field scales with the energy, $\varphi \sim M$.

    For numerical purposes it is useful to rescale cosmological variables according to these scales to avoid variations over many orders of magnitude.
    For example, it is useful to define the dimensionless scalar amplitude $\bar{\varphi} = \varphi/M$, and gauge field $\bar{A}_\mu = A_\mu / H$.
    Ultimately however, the quantities we are interested in are the actual dimensionful quantities $\varphi$, $A_\mu$, etc.

    To facilitate switching between these two unit systems throughout the code, this module provides the classes `BGSystem`, `Val` and `Func`.
    The user may define several variables of interests, e.g., cosmic time, Hubble rate, scalar field amplitude etc. using the `BGVal` class factory, which creates subclasses of `Val`.
    For functions of these variables, like a scalar potential, the same is achieved by the factory `BGFunc`, which creates subclasses of `Func`.
    These objects define a particular scaling with $H_0$ and $M$, e.g., $X = \bar{X} H^a M^b$, where $\bar{X}$ is the re-scaled quantity used for numerical computations.
    These `Func` and `Val` objects are the collected inside a `BGSystem` which defines the common reference scale by setting $H$ and $M$.

    The following examples illustrates the typical workflow using these classes:

    Examples
    --------
    1. Defining a `BGSystem`
    ```python
    from GEFF.bgtypes import BGSystem, BGVal

    # define two variables corresponding to physical time and Hubble rate.
    time = BGVal("t", H0=-1, MP=0)
    Hubble = BGVal("H", H0=1, MP=0)

    # Create a BGSystem with 'time' and 'Hubble'
    # We set the reference frequency to 1e-5*Mpl
    # The reference energy is the Planck mass
    U = BGSystem({time, Hubble}, 1e-5, 1)

    # The BGSystem knows about the new variables time and Hubble
    print(U.quantity_names()) # prints ["t", "H"]

    # However, these variables are not instantiated:
    print(U.value_names()) # prints [] 

    # Instantiate the quantities t and H using the units defined by U:
    U.initialise("t")(1e5)
    U.initialise("H")(1e-5)
    print(U.value_names()) # prints ["t", "H"] 

    # The BGSystem now recognizes "t" and "H" as keys:
    print(f"In Planck units, the Hubble rate at time {U.t} is {U.H}.")
    # gives U.t=1e5, U.H=1e-5

    # Convert everything to numerical units
    U.set_units(False)
    print(f"In numerical units, the Hubble rate at time {U.t} is {U.H}.") 
    # gives U.t=1, U.H=1
    ```

    2. Adding a new `Val` 

    ```python
    #Let us reset the units of U:
    U.set_units(True)

    # add a new Val: the electric-field expectation value E^2
    U.add_BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 

    # initialise E0 with some value in Planck units:
    U.initialise("E0")( 6e-10 )
    ```

    3. Operations between `Val`

    ```python
    # We add a new BGVal to U, the magnetic-field expectation value
    U.add_BGVal("B0", 4, 0)
    U.initialise("B0")(1e-10)

    # We can safely add up E0 and B0 as they are in the same units:
    print(U.E0 + U.B0) #gives 7e-10 = 6e-10 + 1e-10
    ```

    4. Adding a new `Func`

    ```python

    # define a new Func: rhoE, the electric-field energy density
    rhoE = U.add_BGFunc("rhoE", func_args=[U.E0], H0=2, MP=2) # since 3 * M_pl^2 * H^2 = rho
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
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.H0*U.MP)**2 (now in numerical units)

    # Again, this outcome does not depend on the units of E0:
    U.E0.set_units(True)
    print( U.rhoE(U.E0) ) # gives 3. = 3e-10 / (U.H0*U.MP)**2 (in numerical units)
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
    print( U.rhoE(val) ) #gives 3e-20 = 0.5* (6e-10*U.H0**4) / (U.H0*U.MP)**2

    # Unlike a Val, the float does not track units
    # Therefore, it is always assumed to be in the units of rhoE
    # If you want to get the correct result, you would need to convert val by hand:
    print( U.rhoE(val/U.H0**4) ) # gives 3., the expected result in numerical units.

    # Overall, its safer to just keep everything in the same units:
    U.set_units(True)
    ```
    """,

    "BGSystem":"""
    A collection of cosmological variables sharing a system of units.

    Instances of this class define two base unit systems,
    'physical units' and 'numerical units', by setting an energy scale, `MP` and an inverse time scale, `H0`. 

    The cosmological variables (time, Hubble rate etc) are represented by `Quantity` objects.
    These objects are stored in `quantities` and can can be initialise using `initialise`.
    Instances of these objects can be collectively converted between units by using the scales `H0` and `MP`. 

    This class is the fundamental building block of the GEF-code. 
    """,

    "Quantity":r"""
    An object representing a cosmological quantity. 

    A cosmological quantity has a characteristic scaling with respect to a frequency scale (e.g., the Hubble rate at some reference time), and energy scale (e.g. the Planck mass).

    Typical such objects are:
    - cosmic time $t \sim H^{-1}$
    - frequency $f \sim H$
    - scalar-field vev $\varphi \sim M$
    - scalar potential $V(\varphi) \sim (H M)^2$ *(i.e., scales like an energy density)*
    - gauge fields $A_\mu \sim H$ *(i.e., scales like a time derivative)*

    `Quantity` objects are initialized as part of a `BGSystem`, which defines a common frequency scale and energy scale for all its `quantities`.

    This class serves as a parent for `Val` and `Func`.
    """,

    "Val":val_docs+ val_addendum,
    "BGVal.BGVal":val_docs+bgval_addendum,

    "Func":val_docs+ val_addendum,
    "BGFunc.BGFunc":val_docs+bgfunc_addendum

}

