from ._docs import generate_docs, docs_bgtypes
import numpy as np
from typing import Callable, ClassVar
import pandas as pd
from copy import deepcopy

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
        self._omega : float = omega
        self._mu : float = mu
        self._units=True

    @property
    def omega(self) -> float:
        """A frequency scale (typically the Hubble rate at some reference time)"""
        return self._omega
    
    @property 
    def mu(self) -> float:
        """An energy scale (typically the Planck mass)"""
        return self._mu

    @property
    def units(self) -> bool:
        """Indicates the current units of the BGSystem. `True`:physical units, `False`:numerical units"""
        return self._units
    
    @units.setter
    def units(self, newunits:bool):
        """Change the units of the BGSystem and its `Quantity` instances."""
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Quantity):
                obj.units = newunits
        self._units=bool(newunits)
        return

    
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
            units = sys.units
            #match units of new system
            sys.units = True

            #Copy values and functions
            values = sys.variable_list()
            funcs = sys.function_list()
            consts = sys.constant_list()

            for const in consts:
                obj = const
                newinstance.initialise(obj.name)(obj.value)

            for value in values:
                obj = value
                newinstance.initialise(obj.name)(obj.value)

            for func in funcs:
                obj = func
                newinstance.initialise(obj.name)(obj.basefunc)
            
            #restore old units
            sys.units = units
        
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

        return list(self.quantities.keys())
    

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
    
    def variable_list(self) -> list['Variable']:
        """
        Get a list of all `Variable` instances attributed to this BGSystem.

        Returns
        -------
        vals : list of Variable
            the list of `Variable` instances.
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
            the list of `Constant` instances.
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
    
    def add_variable(self, name : str, qu_omega : int, qu_mu : int):
        """
        Define a new `Variable` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        qu_omega : int
            the 'u_omega' parameter of the new object.
        qu_mu : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = define_var(name, qu_omega, qu_mu)
        return
    
    def add_constant(self, name : str, qu_omega : int, qu_mu : int):
        """
        Define a new `Constant` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        qu_omega : int
            the 'u_omega' parameter of the new object.
        qu_mu : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = define_const(name, qu_omega, qu_mu)
        return
    
    def add_function(self, name : str, args : list['Val'], qu_omega : int, qu_mu : int):
        """
        Define a new `Func` object and add it to `quantities`.

        Parameters
        ----------
        name : str
            the name of the new object.
        args : list of BGVal
            the 'args' parameter of the new object.
        qu_omega : int
            the 'u_omega' parameter of the new object.
        qu_mu : int
            the 'u_mu' parameter of the new object.
        """

        self.quantities[name] = define_func(name, args, qu_omega, qu_mu)
        return
    
    def save_variables(self, path : str):
        """
        Save the data in the current GEF instance in an output file.

        Note, data is always stored in numerical units.
        The save will not store constants or functions, only variables.

        Parameters
        ----------
        path : str
            Path to store data in.

        Raises
        ------
        ValueError
            if the GEF object has no data to store.
        """

        storeables = self.variable_names()    
        #Check that all dynamic and derived quantities are initialised in this GEF instance
        if len(storeables) == 0:
            raise ValueError("No data to store.")
        
        #Create a dictionary used to initialise the pandas DataFrame
        dic = {}

        #remember the original units of the GEF
        og_units=self.units

        #Data is always stored unitless
        self.units = False

        for key in storeables:
            dic[key] = getattr(self, key).value
        
        #Create pandas data frame and store the dictionary under the user-specified path
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)

        #after storing data, restore original units
        self.units = og_units
        return
    
    def load_variables(self, path : str):
        """
        Load data and store its results in the BGSystem.

        Note, data is always loaded assuming numerical units.
        Data is only loaded for variables, not for functions or constants.

        Parameters
        ----------
        path : None or str
            Path to load data from

        Raises
        ------
        FileNotFoundError
            if no file is found at `path`.
        AttributeError
            if the file contains a column labeled by a key which does not match any BGSystem variable.
        """
        #Check if file exists
        try:
            #Load from file
            input_df = pd.read_table(path, sep=",")
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found under '{path}'")
        
        #Dictionary for easy access using keys

        data = dict(zip(input_df.columns[1:],input_df.values[:,1:].T))

        #Before starting to load, check that the file is compatible with the GEF setup.
        names = self.quantity_names()
        for key in data.keys():
            if key not in names:
                raise AttributeError(f"The data table you tried to load contains an unknown quantity: '{key}'")
        
        #Store current units to switch back to later
        og_units=self.units

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.units = False
        #Load data into background-value attributes
        for key, values in data.items():
            if key in self.variable_names():
                getattr(self, key).value = values
            else: 
                self.initialise(key)(values)

        self.units = og_units
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

        self._units = sys.units
        self._conversion = (sys.omega**self.u_omega*sys.mu**self.u_mu)

    def __repr__(self):
        r"""A string representing the class, giving its name and scaling with frequency ($\omega$) and energy ($\mu$)."""
        return f"{self.name}({self.u_omega},{self.u_mu})"

    def __str__(self) -> str:
        """The class instance as a string including its name and current units."""

        if not(self._units):
            return f"{self.name} (numerical)"
        elif self._units:
            return f"{self.name} (physical)"
    
    
    @property
    def units(self) -> bool:
        """Indicates the current units of the Quantity. `True`:physical units, `False`:numerical units"""
        return self._units
    
    @units.setter
    def units(self, units : bool):
        """Convert the object between numerical and physical units."""
        self._units = bool(units)
        return
    
    @property
    def conversion(self) -> float:
        """A conversion factor between numerical and physical units."""
        return self._conversion
    
    @classmethod
    def get_description(cls) -> str:
        """Return a string describing the object."""
        if cls.description=="":
            return f"{cls.name}"
        else:
            return f"{cls.name} - {cls.description}"
        

    
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
        self._value =  value*self._conversion**(-sys.units)

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
    
    @property
    def value(self) -> float:
        """The objects value in its respective units."""
        return self._value*self._conversion**self._units
    
    @value.setter
    def value(self, newval):
        """Overwrite the `value` attribute (assuming its current units)."""
        self._value = newval*self._conversion**(-self._units)
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
        
    @property
    def basefunc(self) -> Callable:
        """The underlying function which defines the `__call__` method."""
        return self._basefunc
    
    def get_arg_conversions(self) -> list[float]:
        """
        Get a list of conversion factors for each argument of `basefunc`.

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
        def float_handler(x, i):
            return x*self._arg_conversions[i]**(1-self._units)
        
        def val_handler(x, i):
            conv = x.conversion
            assert self._arg_conversions[i] == conv
            return x*conv**(1-x.units)

        typedic = {Variable : val_handler, Val : val_handler, Constant: val_handler}

        args = [typedic.get(arg.__class__.__bases__[0], float_handler)(arg, i) for i, arg in enumerate(args)]

        return self._basefunc(*args)/self._conversion**(1-self._units)

    
class Variable(Val):
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
    
    @property
    def value(self) -> np.ndarray:
        """The objects value in its respective units."""
        return self._value*self._conversion**(self._units)

    @value.setter
    def value(self, newval):
        """Overwrite the `value` attribute ensuring it is an array"""
        self._value = np.asarray(newval*self._conversion**(-self._units), dtype=self.dtype)
        return
     
class Constant(Val):
    def __init__(self, value, sys : BGSystem):
        """
        Create a new instance using a float and a BGSystem

        Parameters
        ----------
        value : NDArray
            the underlying array of the instance
        sys : BGSystem
            the BGSystem to which the instance belongs

        Raises
        ------
        TypeError 
            if value cannot be converted to a float
        """
        super().__init__( float(value), sys)

class GaugeField:
    name : ClassVar[str] = ""
    """The name of the class."""
    zeros : ClassVar[list[Variable]]= []
    r"""A list of the associated 0$^{\rm th}$ order quantities."""
    cutoff : Variable = None
    """The associated UV cutoff scale."""
        
    @classmethod
    def get_description(cls) -> str:
        """Return a string describing the object."""
        return f"{cls.name} - associated with: {[a.name for a in cls.zeros]}, UV cutoff: {cls.cutoff.name}"
    
def define_var(qname : str, qu_omega : int, qu_mu : int, qdescription:str="", qdtype : np.dtype=np.float64):
    """
    Creates a subclass of `Variable` with custom name, and scaling.

    Parameters
    ----------
    qname : str
        the `name` attribute of the subclass
    qu_omega : int
        the `u_omega` attribute of the subclass
    qu_mu : int
        the `u_mu` attribute of the subclass
    qdescription : str
        a brief description of the subclass
    qdtype : Numpy Data Type
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

    if not( np.issubdtype(qdtype, np.floating) ):
        raise TypeError("BGVal's data-type must be a subtype of 'numpy.floating'.")

    class CustomVar(Variable):
        __doc__ = docs_bgtypes.DOCS["define_var.CustomVar"]
        name=qname
        u_omega = qu_omega
        u_mu = qu_mu
        dtype = qdtype
        description = qdescription
        def __init__(self, value, sys):
            super().__init__(value, sys)
    CustomVar.__qualname__ = f"Val_{qname}"
    CustomVar.__module__ = __name__

    return CustomVar

def define_const(qname : str, qu_omega : int, qu_mu : int, qdescription:str=""):
    """
    Creates a subclass of `Constant` with custom name, and scaling.

    Parameters
    ----------
    qname : str
        the `name` attribute of the subclass
    qu_omega : int
        the `u_omega` attribute of the subclass
    qu_mu : int
        the `u_mu` attribute of the subclass
    qdescription : str
        a brief description of the subclass
        
    Returns
    -------
    CustomConst : class
        the custom subclass
    """

    class CustomConst(Constant):
        __doc__ = docs_bgtypes.DOCS["define_const.CustomConst"]
        name=qname
        u_omega = qu_omega
        u_mu = qu_mu
        description = qdescription
        def __init__(self, value, sys):
            super().__init__(value, sys)
    CustomConst.__qualname__ = f"Const_{qname}"
    CustomConst.__module__ = __name__

    return CustomConst



def define_func(qname : str, func_args : list[Val], qu_omega : int, qu_mu : int, qdescription:str="", qdtype : np.dtype=np.float64):
    """
    Creates a subclass of `Func` with custom name, scaling, and argument signature.

    Parameters
    ----------
    qname : str
        the `name` attribute of the subclass
    qu_omega : int
        the `u_omega` attribute of the subclass
    qu_mu : int
        the `u_mu` attribute of the subclass
    qdtype : Numpy Data Type
        the `dtype` attribute of the subclass
    qdescription : str
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

    if not( np.issubdtype(qdtype, np.floating) ):
        raise TypeError("define_func's data-type must be a subtype of 'np.floating'.")
    
    class CustomFunc(Func):
        __doc__ = docs_bgtypes.DOCS["define_func.CustomFunc"]
        name=qname
        u_omega = qu_omega
        u_mu = qu_mu
        args = func_args
        dtype = qdtype
        description = qdescription
        def __init__(self, func, sys):
            super().__init__(func, sys)

    CustomFunc.__qualname__ = f"Func_{qname}"
    CustomFunc.__module__ = __name__

    return CustomFunc

def define_gauge(qname : str, qzeros : list[Variable], qcutoff : Variable):
    """
    A class factory creating custom  `GaugeField` classes with new name, zero variables, and cutoff scale.

    Parameters
    ----------
    qname : str
        the `name` attribute of the subclass
    qzeros : list of Variable
        the `zeros` attribute of the new subclass 
    qcutoff : Variable
        the `cutoff` attribute of the new subclass

    Returns
    -------
    CustomGaugeField : class
        the custom subclass
    """
    class CustomGaugeField(GaugeField):
        name = qname
        zeros = qzeros
        cutoff = qcutoff
        
    CustomGaugeField.__qualname__ = f"GaugeField_{qname}"
    CustomGaugeField.__module__ = __name__
    return CustomGaugeField

#Add docstrings
generate_docs(docs_bgtypes.DOCS)
    
#Some usful pre-defined quantities
#Space--time variables:
t=define_var("t", -1, 0, "cosmic time")
N=define_var("N", 0, 0, "e-folds")
a=define_var("a", 0, 0, "scale factor")
H=define_var("H", 1, 0, "Hubble rate")

#Inflaton  variables:
phi=define_var("phi", 0, 1, "inflaton expectation value")
dphi=define_var("dphi", 1, 1, "inflaton velocity")
ddphi=define_var("ddphi", 2, 1, "inflaton acceleration")

#Inflaton potential
V=define_func("V", [phi], 2, 2, "scalar potential")
dV=define_func("dV", [phi], 2, 2, "scalar-potential derivative")

#Gauge-field variables:
E=define_var("E", 4, 0, "electric-field expectation value, E^2")
B=define_var("B", 4, 0, "magnetic-field expectation value, B^2")
G=define_var("G", 4, 0, "Chern-Pontryagin expectation value, -E.B")

#Auxiliary quantities:
xi=define_var("xi", 0, 0, "instability parameter")
kh=define_var("kh", 1, 0, "instability scale")

#constants
beta=define_const("beta", 0, -1, "inflaton--gauge-field coupling beta/Mp")

#basic gauge-field
GF = define_gauge("GF", [E, B, G], kh)