from GEFF._docs.docs_bgtypes import (module_docs, bgsystem_docs, quantity_docs, val_docs, val_addendum,
                                        bgval_addendum, func_docs, func_addendum, bgfunc_addendum)
import numpy as np
from copy import deepcopy
from typing import Callable, ClassVar

__doc__ = module_docs

class BGSystem:
    __doc__ = bgsystem_docs

    def __init__(self, quantity_set : set, H0 : float, MP : float):
        """
        Create a new BGSystem in physical units.

        Parameters
        ----------
        quantity_set : set of Quantity
            used to define `quantities`
        H0 : float
            the characteristic frequency scale
        MP : float
            the characteristic energy scale
        """
        self.quantities : dict = {q.name:q for q in quantity_set}
        """A dictionary of all `Quantity` objects for this BGSystem"""
        self.H0 : float = H0
        """A frequency scale (typically the Hubble rate at some reference time)"""
        self.MP : float = MP
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

        newinstance = cls(sys.quantity_set(), sys.H0, sys.MP)

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
            the name of the object
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
    __doc__ = quantity_docs

    name : ClassVar[str]= ""
    """The objects name"""
    u_H0 : ClassVar[int] = 0
    """Indicates how the object scales with frequency."""
    u_MP : ClassVar[int] = 0
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

        if not(self._units):
            return f"{self.name} (numerical)"
        elif self._units:
            return f"{self.name} (physical)"
        
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
    __doc__ = val_docs + val_addendum

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
        super().__init__(sys)
        self.value : np.ndarray =  np.asarray(value, dtype=self.dtype)
        """A 1-D array of values in the units of the class instance."""

    def __str__(self) -> str:
        """
        The class represented as a string

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
    
#CONTINUE FROM HERE!
class Func(Quantity):
    __doc__ = func_docs + func_addendum
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

        self._arg_conversions = [(sys.H0**arg.u_H0*sys.MP**arg.u_MP)
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

        typedic = {Val : val_handler}

        args = [typedic.get(arg.__class__.__bases__[0], float_handler)(arg, i) for i, arg in enumerate(args)]

        return self._basefunc(*args)/self._conversion**(1-units)
    
def BGVal(q_name : str, H0 : int, MP : int, q_dtype : np.floating=np.float64):
    """
    Creates a subclass of `Val` with custom `Val.name`, `Val.u_H0` and `Val.u_MP`.

    Parameters
    ----------
    q_name : str
        the `name` attribute of the subclass
    H0 : int
        the `u_H0` attribute of the subclass
    MP : int
        the `u_MP` attribute of the subclass
    q_dtype : Numpy Data Type
        the `dtype` attribute of the subclass
        
    Returns
    -------
    NewVal : class
        the custom subclass

    Raises
    ------
    TypeError
        if the data type is not a subtype of `numpy.floating`
    """

    if not( np.issubdtype(q_dtype, np.floating) ):
        raise TypeError("BGVal's data-type must be a subtype of 'numpy.floating'.")

    class BGVal(Val):
        __doc__ = val_docs + bgval_addendum

        name=q_name
        u_H0 = H0
        u_MP = MP
        dtype = q_dtype
        def __init__(self, value, sys):
            super().__init__(value, sys)

    return BGVal



def BGFunc(qname : str, func_args : list[Val], H0 : int, MP : int, q_dtype : np.dtype=np.float64):
    """
    Creates a subclass of `Func` with custom `Func.name`, `Func.u_H0` and `Func.u_MP`. and `Func.args`.

    Parameters
    ----------
    q_name : str
        the `name` attribute of the subclass
    H0 : int
        the `u_H0` attribute of the subclass
    MP : int
        the `u_MP` attribute of the subclass
    q_dtype : Numpy Data Type
        the `dtype` attribute of the subclass

    Returns
    -------
    NewFunc : class
        the custom subclass

    Raises
    ------
    TypeError
        if the data type is not a subtype of `numpy.floating`
    """

    if not( np.issubdtype(q_dtype, np.floating) ):
        raise TypeError("BGFunc's data-type must be a subtype of 'np.floating'.")
    
    class BGFunc(Func):
        __doc__ = func_docs + bgfunc_addendum

        name=qname
        u_H0 = H0
        u_MP = MP
        args = func_args
        dtype = q_dtype
        def __init__(self, func, sys):
            super().__init__(func, sys)

    return BGFunc
    
#Space--time variables:
t=BGVal("t", -1, 0) #physical time
N=BGVal("N", 0, 0) #e-folds
a=BGVal("a", 0, 0) #scale factor
H=BGVal("H", 1, 0) #Hubble rate

#Inflaton  variables:
phi=BGVal("phi", 0, 1) #inflaton field
dphi=BGVal("dphi", 1, 1) #inflaton velocity
ddphi=BGVal("ddphi", 2, 1) #inflaton acceleration

#Inflaton potential
V=BGFunc("V", [phi], 2, 2) #scalar potential
dV=BGFunc("dV", [phi], 2, 2) #scalar-potential derivative

#Gauge-field variables:
E=BGVal("E", 4, 0) #electric field expectation value
B=BGVal("B", 4, 0) #magnetic field expectation value
G=BGVal("G", 4, 0) #-EdotB expectation value

#Auxiliary quantities:
xi=BGVal("xi", 0, 0) #instability parameter
kh=BGVal("kh", 1, 0) #instability scale

#constants
beta=BGVal("beta", 0, -1) #inflaton--gauge-field coupling beta/Mp