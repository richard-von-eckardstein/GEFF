from ._docs import generate_docs, docs_bgtypes
import numpy as np
from copy import deepcopy
from typing import Callable, ClassVar

class BGSystem:
    def __init__(self, quantity_set : set, omega : float, mu : float):
        """
        Create a new BGSystem in physical units.

        Parameters
        ----------
        quantity_set : set of Quantity
            used to define `quantities`
        omega : float
            the characteristic frequency scale
        mu : float
            the characteristic energy scale
        """
        self.quantities : dict = {q.name:q for q in quantity_set}
        """A dictionary of all `Quantity` objects for this BGSystem"""
        self.omega : float = omega
        """A frequency scale (typically the Hubble rate at some reference time)"""
        self.mu : float = mu
        """An energy scale(typically the Planck mass)"""
        self._units=True
    
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

        newinstance = cls(sys.quantity_set(), sys.omega, sys.mu)

        if copy:
            #store units of original sys
            units = sys.get_units()
            #match units of new system
            sys.set_units(True)

            #Copy values and functions
            values = sys.variable_list()
            funcs = sys.function_list()
            consts = sys.constant_list()

            for const in consts:
                obj = deepcopy(const)
                newinstance.initialise(obj.name)(obj.value)

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

        def init(obj : np.ndarray | Callable):
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
        self._units=units
        return

    def get_units(self) -> bool:
        """
        Get a boolean indicating the current units of the BGSystem.

        Returns
        -------
        units : bool
            `True`: physical units, `False`:  numerical units.
        """

        return self._units
    
    def variable_list(self) -> list['Variable']:
        """
        Get a list of all `Variable` instances attributed to this BGSystem.

        Returns
        -------
        vals : list of Val
            the list of `Val` instances.
        """

        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Variable):
                vals.append(obj)    
        return vals
    
    def variable_names(self) -> list[str]:
        """
        Get a list of names for all `Variable` instances attributed to this BGSystem.

        Returns
        -------
        names : list of str
            the list of names.
        """

        names = []
        for val in self.variable_list():
            names.append(val.name)
        return names
    
    def constant_list(self) -> list['Constant']:
        """
        Get a list of all `Constant` instances attributed to this BGSystem.

        Returns
        -------
        vals : list of Val
            the list of `Val` instances.
        """

        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Constant):
                vals.append(obj)    
        return vals
    
    def constant_names(self) -> list[str]:
        """
        Get a list of names for all `Constant` instances attributed to this BGSystem.

        Returns
        -------
        names : list of str
            the list of names.
        """

        names = []
        for val in self.constant_list():
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
            the name of the object
        """

        delattr(self, name)
        self.quantities.pop(name)
        return
    
    def add_variable(self, name : str, omega_scaling : int, mu_scaling : int):
        """
        Define a new `Variable` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        omega_scaling : int
            the 'u_omega' parameter of the new object.
        mu_scaling : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = BGVar(name, omega_scaling, mu_scaling)
        return
    
    def add_constant(self, name : str, omega_scaling : int, mu_scaling : int):
        """
        Define a new `Constant` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        omega_scaling : int
            the 'u_omega' parameter of the new object.
        mu_scaling : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = BGConst(name, omega_scaling, mu_scaling)
        return
    
    def add_function(self, name : str, args : list['Val'], omega_scaling : int, mu_scaling : int):
        """
        Define a new `Func` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        args : list of BGVal
            the 'args' parameter of the new object.
        omega_scaling : int
            the 'u_omega' parameter of the new object.
        mu_scaling : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = BGFunc(name, args, omega_scaling, mu_scaling)
        return
    
class Quantity:
    name : ClassVar[str]= ""
    """The objects name"""
    description : ClassVar[str]= ""
    """A brief description of the object"""
    u_omega : ClassVar[int] = 0
    """Indicates how the object scales with frequency."""
    u_mu : ClassVar[int] = 0
    """Indicates how the object scales with energy."""

    def __init__(self, sys : BGSystem):
        """
        Create a new Quantity as part of a BGSystem

        The units of the new object matches those of the BGSystem

        Parameters
        ----------
        sys : BGSystem
            the BGSystem to which the object belongs
        """

        self._units = sys.get_units()
        self._conversion = (sys.omega**self.u_omega*sys.mu**self.u_mu)

    def __repr__(self):
        r"""
        A string representing the class, giving its name and scaling with frequency ($\omega$) and energy ($\mu$).

        Returns
        -------
        repr : str
        """
        return f"{self.name}({self.u_omega},{self.u_mu})"

    def __str__(self) -> str:
        """
        The class instance as a string including its name and current units.

        Returns
        -------
        string : str
        """

        if not(self._units):
            return f"{self.name} (numerical)"
        elif self._units:
            return f"{self.name} (physical)"
        
    def what(self) -> str:
        """
        Return a string describing the object.

        Returns
        -------
        string : str
        """
        return self.description
        
    def get_units(self) -> bool:
        """
        Return a boolean corresponding to the units of the object.

        Returns
        -------
        units : bool
            `True`: physical units, `False`: numerical units.
        """

        return self._units
    
    def set_units(self, units : bool):
        """
        Convert the object between numerical and physical units.

        Parameters
        ----------
        units : bool
            `True`: physical units, `False`: numerical units.
        """

        self._units = units
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
    def __init__(self, value : np.ndarray|float, sys : BGSystem):
        """
        Create a new instance using a BGSystem.

        Parameters
        ----------
        value : NDArray
            the underlying array of the instance
        sys : BGSystem
            the BGSystem to which the instance belongs
        """
        super().__init__(sys)
        self.value =  value
        """The objects value in its respective units."""

    def __str__(self) -> str:
        """
        The class instance as a string including its name, current units, and its value.

        Returns
        -------
        str
            the string representation.
        """

        if not(self._units):
            return f"{self.name} (numerical): {self.value}"
        elif self._units:
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
            self._units=units
            return
        if units and not(self._units):
            self.value *= self._conversion
        elif not(units) and self._units:
            self.value /= self._conversion
        self._units=units
        return
    
    def set_value(self, value : np.ndarray):
        """
        Overwrite the `value` attribute.

        Parameters
        ----------
        value : NDArray or float
            the new value.
        """

        self.value = value
        return
    
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
    

class Func(Quantity):
    args : ClassVar[list[Val]] = []
    """Indicates the argument signature for the class."""
    dtype : ClassVar[np.floating] = np.float64
    """The data type returned by `__call__`."""

    def __init__(self, func : Callable, sys : BGSystem):
        """
        Create a new instance using a `Callable` and a BGSystem

        Parameters
        ----------
        func : NDArray
            the underlying function `f(*args)` of the instance
        sys : BGSystem
            the BGSystem to which the instance belongs

        Raises
        ------
        TypeError
            if the number of arguments of `func` do not match `args`
        ValueError
            if the return of`func` can't be converted to `dtype`
        """
        super().__init__(sys)
        func = np.vectorize(func, otypes=[self.dtype])
        
        try:
            testargs = [1.0 for arg in self.args]
            assert func(*testargs).dtype
        except TypeError:
            raise TypeError("The number of non-default arguments of 'func' needs to match 'len(self.args)'.")
        except ValueError:
            raise ValueError(f"'func' must return a single value which can be converted to '{self.dtype}'")
        
        self._basefunc = func

        self._arg_conversions = [(sys.omega**arg.u_omega*sys.mu**arg.u_mu)
                                 for arg in self.args]
        
    def get_basefunc(self) -> Callable:
        """
        Get the underlying function which defines the `__call__` method.

        Returns
        -------
        basefunc : Callable
            the function
        """
        return self._basefunc
    
    def get_arg_conversions(self) -> list[float]:
        """
        Get a list of conversion factors for each argument of `f(*args)`.

        Returns
        -------
        arg_conversions : list of float
            a list of conversion factors
        """

        return self._arg_conversions
    
    def __call__(self, *args):
        """
        Define the call method of the class as outlined in its documentation.
        """
        units = self.get_units()
        def float_handler(x, i):
            return x*self._arg_conversions[i]**(1-units)
        
        def val_handler(x, i):
            conv = x.get_conversion()
            assert self._arg_conversions[i] == conv
            pow = (1 - x.get_units())
            return x*conv**pow

        typedic = {Variable : val_handler, Val : val_handler, Constant: val_handler}

        args = [typedic.get(arg.__class__.__bases__[0], float_handler)(arg, i) for i, arg in enumerate(args)]

        return self._basefunc(*args)/self._conversion**(1-units)

    
class Variable(Val):
    """A `Val` object representing a variable quantity with time."""

    dtype : ClassVar[np.floating] = np.float64
    """The data type of `value`."""
    
    def __init__(self, value : np.ndarray, sys : BGSystem):
        """
        Create a new instance using a numpy array and a BGSystem

        Parameters
        ----------
        value : NDArray
            the underlying array of the instance
        sys : BGSystem
            the BGSystem to which the instance belongs
        """
        super().__init__( np.asarray(value, dtype=self.dtype), sys)

    #The following methods ensure that a `Val` instance can be used as an array concerning mathematical operations and indexing.
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __len__(self):
        return len(self.value)

    def set_value(self, value : np.ndarray):
        """
        Overwrite the `value` attribute.

        Parameters
        ----------
        value : NDArray or float
            the new value.
        """

        self.value = np.asarray(value)
        return
     
     
     
class Constant(Val):
    """
    A `Val` that represents a constant with time.

    This subclass 
    """
    def __init__(self, value : float, sys : BGSystem):
        """
        Create a new instance using a float and a BGSystem

        Parameters
        ----------
        value : NDArray
            the underlying array of the instance
        sys : BGSystem
            the BGSystem to which the instance belongs
        """
        super().__init__( value, sys)
    
    
def BGVar(qname : str, q_u_omega : int, q_u_mu : int, q_description:str="", q_dtype : np.dtype=np.float64):
    """
    Creates a subclass of `Variable` with custom name, and scaling.

    Parameters
    ----------
    q_name : str
        the `name` attribute of the subclass
    q_u_omega : int
        the `u_omega` attribute of the subclass
    q_u_mu : int
        the `u_mu` attribute of the subclass
    q_description : str
        a brief description of the subclass
    q_dtype : Numpy Data Type
        the `dtype` attribute of the subclass
    
        
    Returns
    -------
    CustomVar : class
        the custom subclass

    Raises
    ------
    TypeError
        if the data type is not a subtype of `numpy.floating`
    """

    if not( np.issubdtype(q_dtype, np.floating) ):
        raise TypeError("BGVal's data-type must be a subtype of 'numpy.floating'.")

    class CustomVar(Variable):
        __doc__ = docs_bgtypes.DOCS["BGVar.CustomVar"]
        name=qname
        u_omega = q_u_omega
        u_mu = q_u_mu
        dtype = q_dtype
        description = q_description
        def __init__(self, value, sys):
            super().__init__(value, sys)
    CustomVar.__qualname__ = f"Val_{qname}"
    CustomVar.__module__ = __name__

    return CustomVar

def BGConst(qname : str, q_u_omega : int, q_u_mu : int, q_description:str=""):
    """
    Creates a subclass of `Constant` with custom name, and scaling.

    Parameters
    ----------
    q_name : str
        the `name` attribute of the subclass
    q_u_omega : int
        the `u_omega` attribute of the subclass
    q_u_mu : int
        the `u_mu` attribute of the subclass
    q_description : str
        a brief description of the subclass
        
    Returns
    -------
    CustomConst : class
        the custom subclass
    """

    class CustomConst(Constant):
        __doc__ = docs_bgtypes.DOCS["BGConst.CustomConst"]
        name=qname
        u_omega = q_u_omega
        u_mu = q_u_mu
        description = q_description
        def __init__(self, value, sys):
            super().__init__(value, sys)
    CustomConst.__qualname__ = f"Const_{qname}"
    CustomConst.__module__ = __name__

    return CustomConst



def BGFunc(qname : str, func_args : list[Val], q_u_omega : int, q_u_mu : int, q_description:str="", q_dtype : np.dtype=np.float64):
    """
    Creates a subclass of `Func` with custom name, scaling, and argument signature.

    Parameters
    ----------
    q_name : str
        the `name` attribute of the subclass
    q_u_omega : int
        the `u_omega` attribute of the subclass
    q_u_mu : int
        the `u_mu` attribute of the subclass
    q_dtype : Numpy Data Type
        the `dtype` attribute of the subclass
    q_description : str
        a brief description of the subclass

    Returns
    -------
    CustomFunc : class
        the custom subclass

    Raises
    ------
    TypeError
        if the data type is not a subtype of `numpy.floating`
    """

    if not( np.issubdtype(q_dtype, np.floating) ):
        raise TypeError("BGFunc's data-type must be a subtype of 'np.floating'.")
    
    class CustomFunc(Func):
        __doc__ = docs_bgtypes.DOCS["BGFunc.CustomFunc"]
        name=qname
        u_omega = q_u_omega
        u_mu = q_u_mu
        args = func_args
        dtype = q_dtype
        description = q_description
        def __init__(self, func, sys):
            super().__init__(func, sys)

    CustomFunc.__qualname__ = f"Func_{qname}"
    CustomFunc.__module__ = __name__

    return CustomFunc


#Add docstrings
generate_docs(docs_bgtypes.DOCS)
    
#Some usful pre-defined quantities
#Space--time variables:
t=BGVar("t", -1, 0, "physical time")
N=BGVar("N", 0, 0, "e-folds")
a=BGVar("a", 0, 0, "scale factor")
H=BGVar("H", 1, 0, "Hubble rate")

#Inflaton  variables:
phi=BGVar("phi", 0, 1, "inflaton expectation value")
dphi=BGVar("dphi", 1, 1, "inflaton velocity")
ddphi=BGVar("ddphi", 2, 1, "inflaton acceleration")

#Inflaton potential
V=BGFunc("V", [phi], 2, 2, "scalar potential")
dV=BGFunc("dV", [phi], 2, 2, "scalar-potential derivative")

#Gauge-field variables:
E=BGVar("E", 4, 0, "electric field expectation value, E^2")
B=BGVar("B", 4, 0, "magnetic field expectation value, B^2")
G=BGVar("G", 4, 0, "Chern-Pontryagin expectation value, -E.B")

#Auxiliary quantities:
xi=BGVar("xi", 0, 0, "instability parameter")
kh=BGVar("kh", 1, 0, "instability scale")

#constants
beta=BGConst("beta", 0, -1, "inflaton--gauge-field coupling beta/Mp")