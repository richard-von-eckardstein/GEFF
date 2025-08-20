import numpy as np
from copy import deepcopy
from numpy.typing import NDArray
from typing import Callable


class BGSystem:
    """
    A collection of cosmological variables sharing a system of units.

    Instances of this class define two base unit systems,
    'physical units' and 'numerical units', by setting two scales:
    - **H0**: a frequency scale (typically the Hubble rate at some reference time)
    - **MP**: an energy scale (typically the Planck mass)

    The cosmological variables (time, Hubble rate etc) are represented by `Quantity` objects.
    Instances of these objects can be converted between units
    by using the scales H0 and MP.  
    This class is the fundamental building block of the GEF-code. 

    Attributes
    ----------
    H0 : float
        the frequency scale
    MP : float
        the energy scale
    quantities : dict
        A dictionary of all `Quantity` objects for this BGSystem
    'QuantityName' : Val or Func
        Instances of `Quantity`

    Example
    -------
    ```python
    from GEFF.bgtypes import BGVal, BGSystem

    # define two variables corresponding to physical time and Hubble rate.
    time = BGVal("t", H0=-1, MP=0)
    Hubble = BGVal("H", H0=1, MP=0)

    # Create a BGSystem with 'time' and 'Hubble' and the reference frequency set to 1e-5*Mpl
    U = BGSystem({time, Hubble}, 1e-5, 1)

    #The BGSystem knows about the new variables time and Hubble
    print(U.object_names()) #prints ["t", "H"]

    #However, these variables are not instantiated:
    print(U.value_names()) #prints [] 
    
    #Instantiate the quantities 't' and 'H' using the units defined by 'U':
    U.initialise("t")(1e5)
    U.initialise("H")(1e-5)
    print(U.value_names()) #prints ["t", "H"] 
    
    #The BGSystem now recognizes "t" and "H" as keys:
    print(f"In Planck units, the Hubble rate at time {U.t} is {U.H}.")
    # gives U.t=1e5, U.H=1e-5
    
    #Convert everything to numerical units
    U.set_units(False)
    print(f"In numerical units, the Hubble rate at time {U.t} is {U.H}.")
    # gives U.t=1, U.H=1
    ```
    """

    def __init__(self, quantity_set : set, H0 : float, MP : float):
        self.quantities = {q.name:q for q in quantity_set}
        self.H0 = H0
        self.MP = MP
        self.__units=True
    
    @classmethod
    def from_system(cls, sys : 'BGSystem', copy : bool=False) -> 'BGSystem':
        """Initialize a new `BGSystem` from an existing instance.

        The new instance is created with the same quantities, reference frequency and amplitude as the original.
        If specified, the `Quantity` instances are also copied to the new BGSystem

        Parameters
        ----------
        sys : BGSystem
            the original instance used as a template
        copy : Boolean
            `True` if `Quantity` instances are also copied
        Returns
        -------
        newinstance : BGSystem
            the new instance
        """

        newinstance = cls(sys.object_set(), sys.H0, sys.MP)

        if copy:
            #store units of original sys
            units = sys.get_units()
            #match units of new system
            sys.set_units(True)

            #Copy values and functions
            values = sys.value_list()
            funcs = sys.function_list()

            for value in values:
                obj = deepcopy(value)
                newinstance.initialise(obj.name)(obj.value)

            for func in funcs:
                obj = deepcopy(func)
                newinstance.initialise(obj.name)(obj.get_basefunc())
            
            #restore old units
            sys.set_units(units)
        
        return newinstance
    
    def quantity_set(self) -> set[object]:
        """
        Get a set of all `Quantity` objects attributed to this BGSystem.

        Returns
        -------
        set : set
            a set of objects.
        """

        return set(self.quantities.values())
    
    def quantity_names(self) -> list[str]:
        """
        Get a list of names for all `Quantity` objects attributed to this BGSystem.

        Returns
        -------
        names : list of str
            the list of names.
        """

        return self.quantities.keys()
    

    def initialise(self, quantity : str) -> Callable:
        """
        Instantiate a `Quantity` object from `quantities`.

        The method creates a function `init` which can be called by
        an arithmetic type / `Callable` to instantiate a `Val` / `Func`.
        Calling `init` adds and instance of the `Quantity` as a new attribute to the BGSystem,
        with the attribute name corresponding to the object's `name` attribute.   

        Parameters
        ----------
        quantity : str
            the name of the object which is to be instantiated.

        Returns
        -------
        init : Callable
            a function used to initialize the `Quantity` object
        """

        def init(obj : NDArray | Callable):
            """
            Initialize a `Quantity` object with an arithmetic type / Callable.

            This adds an instance of the `Quantity` as a new attribute to the `BGSystem`, with the attribute name
            corresponding to the `Quantity` object's `name` attribute. 

            Parameters
            ----------
            obj : NDArray or Callable
                the NDArray / Callable with which the `Quantity` is to be instantiated.
            """

            q = self.quantities[quantity]
            setattr( self, quantity, q(obj, self) )
            return
        
        return init
        
    def set_units(self, units : bool):
        """
        Change the units of the BGSystem and its `Quantity` instances.

        Parameters
        ----------
        units : bool
            `True`: physical units, `False`:  numerical units.
        """

        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Quantity):
                obj.set_units(units)
        self.__units=units
        return

    def get_units(self) -> bool:
        """
        Get a boolean indicating the current units of the BGSystem.

        Returns
        -------
        units : bool
            `True`: physical units, `False`:  numerical units.
        """

        return self.__units
    
    def value_list(self) -> list['Val']:
        """
        Get a list of all `Val` instances attributed to this BGSystem.

        Returns
        -------
        vals : list of Val
            the list of `Val` instances.
        """

        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Val):
                vals.append(obj)    
        return vals
    
    def value_names(self) -> list[str]:
        """
        Get a list of names for all `Val` instances attributed to this BGSystem.

        Returns
        -------
        names : list of str
            the list of names.
        """

        names = []
        for val in self.value_list():
            names.append(val.name)
        return names

    def function_list(self) -> list['Func']:
        """
        Get a list of all `Func` instances attributed to this BGSystem.

        Returns
        -------
        funcs : list of Func
            the list of `Func` instances.
        """

        funcs = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Func):
                    funcs.append(obj)      
        return funcs
    
    def function_names(self) -> list[str]:
        """
        Get a list of names for all `Func` instances attributed to this BGSystem.

        Returns
        -------
        names : list of str
            the list of names.
        """

        names = []
        for val in self.function_list():
            names.append(val.name)
        return names
    
    def remove(self, name : str):
        """
        Remove a `Quantity` object and its instance from the BGSystem.

        Parameters
        ----------
        name : str
            the name of the BGVal / BGFunc object
        """

        delattr(self, name)
        self.quantities.pop(name)
        return
    
    def add_BGVal(self, name : str, H0units : int, MPunits : int):
        """
        Define a new `Val` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        H0units : int
            the 'u_H0' parameter of the new object.
        MPunits : int
            the 'u_MP' parameter of the new object.
        """

        self.quantities[name] = BGVal(name, H0units, MPunits)
        return
    
    def add_BGFunc(self, name : str, args : list['Val'], H0units : int, MPunits : int):
        """
        Define a new `Func` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        args : list of BGVal
            the 'args' parameter of the new object.
        H0units : int
            the 'u_H0' parameter of the new object.
        MPunits : int
            the 'u_MP' parameter of the new object.
        """

        self.quantities[name] = BGFunc(name, args, H0units, MPunits)
        return
    
class Quantity:
    r"""
    An object representing a cosmological quantity. 

    A cosmological quantity has a characteristic scaling with respect to a frequency scale (e.g., the Hubble rate at some reference time), and energy scale (e.g. the Planck mass).

    Typical such objects are:
    - cosmic time $t$: *scales with* $H^{-1}$
    - frequency $f$: *scales with* $H$
    - scalar-field vev $\varphi$: *scales with* $M_{\rm P}$
    - scalar potential $V(\varphi)$: *scales with* $(H M_{\rm P})^2$ *(i.e. like an energy density)*
    - gauge fields $A_\mu$: *scales with* $H$ *(i.e. like a time derivative)*

    `Quantity` objects are initialized as part of a `BGSystem`, which defines a common frequency scale and energy scale for all its `quantities`.

    This class serves as a parent for `Val` and `Func`.
    
    Attributes
    ----------
    name : str
        the objects name
    u_H0 : int
        an integer indicating how the object scales with frequency
    u_MP : int
        an integer indicating how the object scales with energy
    dtype : Numpy Data Type
        the data type of the object.
    """

    name= ""
    u_H0 = 0
    u_MP = 0
    dtype = np.float64

    def __init__(self, sys : BGSystem):
        self.__units = sys.get_units()
        self._conversion = (sys.H0**self.u_H0*sys.MP**self.u_MP)

    def __repr__(self):
        """
        The class represented as a string stating its scaling with 'H0' and 'MP'.

        Returns
        -------
        repr : str
            the string representation.
        """
        return f"{self.name}(H0={self.u_H0}, MP={self.u_MP})"

    def __str__(self) -> str:
        """
        The class instance represented as a string including its name and current units.

        Returns
        -------
        string : str
            the string representation.
        """

        if not(self.__units):
            return f"{self.name} (numerical)"
        elif self.__units:
            return f"{self.name} (physical)"
        
    def get_units(self) -> bool:
        """
        Return a boolean corresponding to the units of the object.

        Returns
        -------
        units : bool
            `True`: physical units, `False`: numerical units.
        """

        return self.__units
    
    def set_units(self, units : bool):
        """
        Convert the object between numerical and physical units.

        Parameters
        ----------
        units : bool
            `True`: physical units, `False`: numerical units.
        """

        self.__units = units
        return
    
    def get_conversion(self) -> float:
        """
        Get the objects conversion factor between numerical and physical units.

        Returns
        -------
        conversion : float
            the conversion factor
        """

        return self._conversion

    
class Val(Quantity):
    """
    A `Quantity` representing real-valued variables like cosmic time or Hubble rate.
     
    Instances of this object can be used like a 1-D Numpy-Array for mathematical operations and indexing.
    This class defines basic arithmetic operations for its instances as operations on their underlying 1-D Numpy-Array, `value`.  
    This subclass inherits all attributes and methods from `Quantity`, but re-defines the `set_units` method
    such that changing units converts the value of the underlying `value` according to its conversion factor.  
    To define a custom `Val` object with its own name and scaling, use the class factory `BGVal` (see the example below).

    Attributes
    ----------
    name : str
        the objects name
    u_H0 : int
        an integer indicating how the object scales with frequency
    u_MP : int
        an integer indicating how the object scales with energy
    value : NDArray
        a 1-D array of values in the units of this `Val` instance.
    dtype : Numpy Data Type
        the data type of `value`
    
    Examples
    --------
    1. *Defining a `Val`*
    ```python
    from GEFF.bgtypes import BGSystem, BGVals
    # create a BGVal object: the electric field expectation value E^2
    E0 = BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 

    # Add 'E0' to a BGSystem in Planck units (M_pl = 1) and H0 = 1e-5*M_pl:
    U = BGSystem({E0}, H0=1e-5, MP=1.)

    # initialise E0 with some value in Planck units:
    U.initialise("E0")( 6e-10 )

    # U.E0 is now an instance of BGVal:
    print(U.value_names()) # gives ["E0"]
    ```
    2. *unit conversion*
    ```python
    ...
    # Calling in Planck units:
    print( U.E0.value ) # gives 6e-10 (in Planck units)

    # Switch E0 to numerical units by changing the units of U:
    U.set_units(False)
    print(U.E0.value) # gives 6e10 = 6e-10 * U.H0**4 (in numerical units)
    ```
    3. *Adding `Val`'s*
    ```python
    # We add a new BGVal to 'U', the expectation value of B^2
    U.add_BGVal("B0", H0=4, MP=0)

    # Convert back to physical units and initialise it. 
    U.set_units(True)
    U.initialise("B0")(1e-10)

    #We can safely add up E0 and B0 as U ensures they are in the same units:
    print(U.E0 + U.B0) #gives 7e-10 = 6e-10 + 1e-10
    ```
    """
    def __init__(self, value : NDArray, sys : BGSystem):
        super().__init__(sys)
        self.value = np.asarray(value, dtype=self.dtype)

    def __str__(self) -> str:
        """
        The class represented as a string

        Returns
        -------
        str
            the string representation.
        """

        if not(self.__units):
            return f"{self.name} (numerical): {self.value}"
        elif self.__units:
            return f"{self.name} (physical): {self.value}"
    
    def set_units(self, units : bool):
        """
        Convert the object between numerical and physical units.

        Parameters
        ----------
        units : bool
            `True`: physical units, `False`: numerical units.
        """

        if isinstance(self.value, type(None)):
            self.__units=units
            return
        if units and not(self.__units):
            self.value *= self._conversion
        elif not(units) and self.__units:
            self.value /= self._conversion
        self.__units=units
        return
    
    def set_value(self, value : NDArray):
        """
        Overwrite the `value` attribute.

        Parameters
        ----------
        value : NDarray or float
            the new value.
        """

        self.value = np.asarray(value)
        return
    
    #The following methods ensure that a `Val` instance can be used as an array concerning mathematical operations and indexing.
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __len__(self):
        return len(self.value)
    
    #Define all mathematical operations as operations acting on self.value
    def __abs__(self):
        return abs(self.value)
    
    def __neg__(self):
        return -self.value
    
    def __pos__(self):
        return +self.value
    
    def __add__(self, other):
        #self.__Compatible(self, other, "+")
        return self.value + other
        
    __radd__ = __add__
    
    def __sub__(self, other):
        #self.__Compatible(self, other, "-")
        return self.value - other
        
    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other
        
    __rmul__ = __mul__
    
    def __floordiv__(self, other):
        return self.value // other
        
    def __rfloordiv__(self, other):
        return other // self.value
    
    def __truediv__(self, other):
        return self.value / other
        
    def __rtruediv__(self, other):
        return other / self.value
    
    def __mod__(self, other):
        return self.value % other
    
    def __pow__(self, other):
        #BGVal should never be exponentiated by another BGVal
        return self.value ** other
    
    def __eq__(self, other):
        #self.__Compatible(self, other, "==")
        return self.value == other
            
    def __ne__(self, other):
        #self.__Compatible(self, other, "!=")
        return self.value != other
    
    def __lt__(self, other):
        #self.__Compatible(self, other, "<")
        return self.value < other

    def __gt__(self, other):
        #self.__Compatible(self, other, ">")
        return self.value > other

    def __le__(self, other):
        #self.__Compatible(self, other, "<=")
        return  self.value <= other

    def __ge__(self, other):
        #self.__Compatible(self, other, ">=")
        return  self.value >= other
    

def BGVal(qname : str, H0 : int, MP : int, qdtype : np.dtype=np.float64):
    """
    Creates a subclass of `Val` with custom values for `name`, `u_H0` and `u_MP`.

    Parameters
    ----------
    name : str
        the `name` attribute of the subclass
    H0 : int
        the `u_H0` attribute of the subclass
    MP : int
        the `u_MP` attribute of the subclass
    dtype : Numpy Data Type
        the `dtype` attribute of the subclass
        
    Returns
    -------
    BGVal
        a subclass of `Val`.

    Raises
    ------
    TypeError
        if the data type is not a subtype of 'numpy.floating'
    """

    if not( np.issubdtype(qdtype, np.floating)
             or
             np.issubdtype(qdtype, np.complexfloating)
            ):
        raise TypeError("BGVal's data-type must be a subtype of 'numpy.floating' or 'numpy.complexfloating'.")

    class BGVal(Val):
        """
        This class represents a variable with a given mass dimension and scaling behavior with respect to a characteristic frequency (Hubble rate) and energy scale (Planck mass).

        It inherits arithmetic operations and methods for unit conversion from `Val`. Therefore, a 'BGVal' can be used as a 1D Numpy-array regarding indexing and mathematical operations.
        Furthermore, the value of the underlying 1D Numpy-array can be converted from physical to numerical units and vice-versa when instantiated as part of a `BGSystem`.
        Typical 'BGVal' objects are:
            - time: scales with H^(-1), M_pl^0
            - frequency: scales with H^1, M_pl^0
            - scalar-field amplitude: scales with H^0, M_pl^1
            - gauge-fields: scales with H^1, M_pl^0 (i.e. like a time derivative)
    
        Attributes
        ----------
        name : str
            the quantities' name
        u_H0 : int
            an integer indicating how the 'BGVal' scales with frequency
        u_MP : int
            an integer indicating how the 'BGVal' scales with energy
        dtype : type
            the data type of the 'BGVal' object.
        value : NDArray
            a 1-dimensional array of values set to the current units of the 'BGVal' instance.

        Methods
        -------
        get_units()
            Return a boolean indicating if the 'BGVal' instance is in numerical of physical units.
        set_units()
            Convert the 'BGVal' instance between numerical and physical units by converting 'self.value'.
        get_conversion()
            Get the conversion factor between numerical and physical units for the 'BGVal' instance.
        set_value()
            Overwrite 'self.value'

        Example
        -------
        Defining a BGVal:
        ```python
        from GEFF.bgtypes import BGSystem, BGVals
        #create a BGVal object: the electric field expectation value E^2
        E0 = BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 

        #Add 'E0' to a BGSystem in Planck units (M_pl = 1) and H0 = 1e-5*M_pl:
        U = BGSystem({E0}, H0=1e-5, MP=1.)

        #initialise E0 with some value in Planck units:
        U.initialise("E0")( 6e-10 )

        #U.E0 is now an instance of BGVal:
        print(U.value_names()) # gives ["E0"]
        ```
        ... 
        >>> 
        ... 
        ... 
        >>> U = BGSystem({E0}, H0=1e-5, MP=1.)
        ... 
        ... #initialise E0 with some value in Planck units:
        >>> U.initialise("E0")( 6e-10 )
        ... #U.E0 is now an instance of BGVal

        Example 2 (converting the units of a BGVal)
        -------------------------------------------
        ... #Calling in Planck units is straight forwards:
        >>> print( U.E0.value )#gives 6e-10 (in Planck units)
        ... 
        ... #Switch E0 to numerical units changes the value stored in U.E0:
        >>> U.E0.set_units(False)
        >>> print( U.E0.value )#gives 6e10 = 6e-10 * U.H0**4 (in numerical units)

        Example 3 (addition of BGVal's)
        -------------------------------
        ... #We add a new BGVal to 'U', the expectation value of B^2
        >>> U.add_BGVal("B0", H0=4, MP=0)
        ...
        ... #Convert back to physical units and initialise it.
        >>> U.set_units(True)
        >>> U.initialise("B0")(1e-10)
        ...
        ... #We can safely add up E0 and B0 as they are in the same units:
        >>> print(U.E0 + U.B0) #gives 7e-10 = 6e-10 + 1e-10
        """

        name=qname
        u_H0 = H0
        u_MP = MP
        dtype = qdtype
        def __init__(self, value, sys):
            super().__init__(value, sys)

    return BGVal

#CONTINUE FROM HERE!
class Func(Quantity):
    """
    The parent class for 'BGFunc', representing real-valued functions of variables like the inflaton potential.
    
    A `Func` object inherits all attributes and methods from 'Quantities'.
    An instance of `Func` can be used like a function, evaluating the underlying Callable method differently depending on the current units.
    When in physical units, a call to a `Func` returns the result of the underlying function, f(*args).
    When in numerical units, the call instead returns f(*args) divided by the conversion factor defined via its parent `Quantity`.
    If called by a `Val` object, the argument is also converted according to the units of the `Val` instance (generically, identical to the ones of the `Func` instance).
    If called by a regular arithmetic data type (e.g., 'float'), it is assumed that the argument is in the same unit system as the `Func` instance.

    Attributes
    ----------
    args : list of `Val`
        a list of `Val` subclasses indicating how the individual arguments of `Func` scale with frequency and energy.
    
    Methods
    -------
    get_basefunc()
        Get the underlying function used to define the __call__ method.
    get_units()
        A boolean corresponding to the current units of the `Func` instance.
    set_units()
        Convert the Func between numerical and physical units.
    get_conversion()
        Get the conversion factor between numerical and physical units for this `Func` instance.
    """
    args = []
    def __init__(self, func, sys):
        super().__init__(sys)
        func = np.vectorize(func, otypes=[self.dtype])
        
        try:
            testargs = [1.0 for arg in self.args]
            assert func(*testargs).dtype
        except TypeError:
            raise TypeError("The number of non-default arguments of 'func' needs to match 'len(self.args)'.")
        except ValueError:
            raise ValueError("'func' must return a single value which can be converted to '{self.dtype}'")
        
        self._basefunc = func

        self._arg_conversions = [(sys.H0**arg.u_H0*sys.MP**arg.u_MP)
                                 for arg in self.args]
        
    def get_basefunc(self) -> Callable:
        """
        Get the underlying function defining the __call__ method.

        Returns
        -------
        Callable:
            the function.
        """
        return self._basefunc
    
    def get_arg_conversions(self) -> list[float]:
        """
        Get a list of conversion factors between numerical and physical units for each argument.

        Returns
        -------
        list of float
            the list of conversion factors
        """

        return self._arg_conversions
    
    #the dunder method defines the call signature as described in the class documentation
    def __call__(self, *args):
        units = self.get_units()
        def float_handler(x, i):
            return x*self._arg_conversions[i]**(1-units)
        
        def val_handler(x, i):
            conv = x.get_conversion()
            assert self._arg_conversions[i] == conv
            pow = (1 - x.get_units())
            return x*conv**pow

        typedic = {Val : val_handler}

        args = [typedic.get(arg.__class__.__bases__[0], float_handler)(arg, i) for i, arg in enumerate(args)]

        return self._basefunc(*args)/self._conversion**(1-units)


def BGFunc(qname : str, func_args : list[Val], H0 : int, MP : int, qdtype : np.dtype=np.float64):
    """
    A class factory used to define a subclass of `Func` with user-specified values for 'name', 'u_H0', 'u_MP' and 'args'.

    Parameters
    ----------
    name : str
        the 'name' attribute of the new subclass
    func_args : list of `Val`
        the 'args' attribute of the new subclass
    H0 : int
        the 'u_H0' attribute of the new subclass
    MP : int
        the 'u_MP' attribute of the new subclass
    dtype : Numpy Data Type
        the 'dtype' attribute of the new subclass

    Returns
    -------
    BGFunc
        a subclass of `Func`
    """

    if not( np.issubdtype(qdtype, np.floating)
             or
             np.issubdtype(qdtype, np.complexfloating)
            ):
        raise TypeError("BGFunc's data-type must be a subtype of np.floating or np.complexfloating.")
    
    class BGFunc(Func):
        """
        This class represents a dimensionful function of cosmological quantities with a given mass dimension and scaling w.r.t.
        the Hubble rate and Planck mass.
        Typical functions are:
            - scalar potential: scales with H^2, M_pl^2, is a function of a scalar-field amplitude
            - scalar--gauge-field coupling: scales with H^0, M_pl^1, is a function of a scalar-field amplitude
        BGFuncs inherit from `Func` including __call_ and methods related to unit conversion. 

        A call to a BGFunc (inherited from `Func`) returns the result of the underlying basefunction, f(*args), according to f( *args (in physical units) ).
        The result of this operation are returned in numerical / physical units depending on the units of the current BGFunc-instance.
        If called by a BGVal, the conversion of the argument is done using the units of the BGVal-instance. 
        If called by a non-BGVal arithmetic data type, it is assumed that the argument is in the same unit system as the BGFunc instance. 

        Attributes
        ----------
        name : str
            the quantities name
        args : list of BGVal
            a list of BGVals indicating the arguments of the BGFunc.
        u_H0 : int
            the quantities' scaling w.r.t. a typical inverse time scale
        u_MP : int
            the quantities' scaling w.r.t. a typical mass scale
        dtype : Numpy Data Type
            the data type of the BGVal object.

        Methods
        -------
        get_basefunc()
            (inherited from `Func`) returns the underlying function defining the BGFunc-instance.
        get_units()
            (inherited from `Func`) returns `True` if the BGFunc-instance is set to physical units and `False` if set to numerical units.
        set_units()
            (inherited from `Func`) change __call__() of the BGFunc-instance to physical units or numerical units.
        get_conversion()
            (inherited from `Func`) retrieve the conversion factor for this BGFunc-instance
        
        Example 1 (defining a BGFunc) 
        -----------------------------
        ... #create a BGVal object: the electric field expectation value E^2
        >>> E0 = BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 
        ...
        ... #define a BGFunc object: rhoE, the electric field energy density
        >>> rhoE = BGFunc("rhoE", args=[E0], H0=2, MP=2) # since 3 * M_pl^2 * H^2 = rho
        ... 
        ... #collect both in a BGSystem initialised in Planck units (M_pl = 1) and H0 = 1e-5*M_pl:
        >>> U = BGSystem({E0, rhoE}, H0=1e-5, MP=1.)
        ... 
        ... #initialise E0 with some value in Planck units:
        >>> U.initialise("E0")( 6e-10 )
        ... #define a function: rhoE = 0.5*E^2
        >>> func = lambda x: 0.5 * x
        >>> U.initialise("rhoE")( func )
        ... 
        ... #U.rhoE is now a Callable function with a single argument

        Example 2 (calling a BGFunc by a BGVal)
        ---------------------------------------
        ... #Calling in Planck units is straight forwards:
        >>> print( U.rhoE( U.E0 ) )#gives 3e-10 (in Planck units)
        ... 
        ... #compare this to a direct call to 'func':
        >>> print( func(6e-10) )  #gives 3e-10 (in Planck units)
        ... 
        ... #Even if we switch E0 to numerical units, as long as rhoE is in physical units, we get the same result:
        >>> U.E0.set_units(False) #E0.value = 6e10
        >>> print( U.rhoE(U.E0) ) #gives 3e-10 (still in Planck units)
        ... 
        ... #Switching U.rhoE to numerical units means calling rhoE returns values in numerical units:
        >>> U.rhoE.set_units(False)
        >>> print( U.rhoE(U.E0) ) #gives 3. = 3e-10 / (U.H0*U.MP)**2 (in numerical units)
        ... 
        ... #Again, this outcome does not depend on the units of E0:
        >>> U.E0.set_units(True)
        >>> print( U.rhoE(U.E0) ) #gives 3. = 3e-10 / (U.H0*U.MP)**2 (in numerical units)

        Example 3 (calling a BGFunc by a float)
        ---------------------------------------
        ... #instead of calling rhoE by E0, we can call it by a float:
        >>> val = 6e-10
        ... 
        ... #First the behaviour if rhoE is in physical units:
        >>> U.rhoE.set_units(True)
        >>> print( U.rhoE( val ) )  #gives 3e-10 (in Planck units)
        ... 
        ... #Things are different if rhoE is in numerical units:
        >>> U.rhoE.set_units(False)
        >>> print( U.rhoE(val) ) #gives 3e-20 = 0.5* (6e-10*U.H0**4) / (U.H0*U.MP)**2
        ... 
        ... #since val does not have units, it is assumed to be in the units of rhoE.
        ... # If you want this to give the correct result, you would need to convert val by hand:
        >>> print( U.rhoE(val/U.H0**4) ) #gives 3., the expected result in numerical units. 

        """

        name=qname
        u_H0 = H0
        u_MP = MP
        args = func_args
        dtype = qdtype
        def __init__(self, func, sys):
            super().__init__(func, sys)

    return BGFunc
    
