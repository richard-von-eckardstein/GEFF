import numpy as np
from copy import deepcopy
import inspect

from numpy.typing import ArrayLike

def BGVal(Qname : str, H0 : int, MP : int, Qdtype : np.dtype=np.float64):
    """
    A class factory used to define cosmological quantities like "time", "Hubble rate" etc. with a specific scaling behaviour w.r.t
    an inverse time scale (typically Hubble rate) and a mass scale (typically the Planck mass).

    Parameters
    ----------
    name : str
        the quantities name
    H0 : int
        the quantities' scaling w.r.t. a typical inverse time scale
    MP : int
        the quantities' scaling w.r.t. a typical mass scale
    dtype : Numpy Data Type
        the data type of the BGVal object.
        
    Returns
    -------
    BGVal
        a class representing a cosmological quantity like physical time, Hubble rate, etc.
    """

    if not( np.issubdtype(Qdtype, np.floating)
             or
             np.issubdtype(Qdtype, np.complexfloating)
            ):
        raise TypeError("BGVal's data-type must be a subtype of np.floating or np.complexfloating.")

    class BGVal(Val):
        """
        This class represents a cosmological quantity with a given mass dimension and scaling w.r.t. the Hubble rate and Planck mass.
        Typical quantities are:
            - time: scales with H^(-1), M_pl^0
            - time derivative: scales with H^1, M_pl^0
            - scalar-field amplitude: scales with H^0, M_pl^1
            - scalar-field potential: scales with H^2 M_pl^2
            - gauge-fields: scales with H^1, M_pl^0 (i.e. like a time derivative)
        This class inherits from 'Val' including its arithemtic operations and methods relating to unit conversion.

        Attributes
        ----------
        name : str
            the quantities name
        u_H0 : int
            the quantities' scaling w.r.t. a typical inverse time scale
        u_MP : int
            the quantities' scaling w.r.t. a typical mass scale
        dtype : Numpy Data Type
            the data type of the BGVal object.
        value : NDArray
            (inherited from 'Val') a 1-dimensional array of values set to the current units of this BGVal-instance.
            This attribute is used for all arithmetic operations and is converted to physical and numerical units via SetUnits().
        massdim : int
            (inherited from 'Val') the mass dimension of this BGVal-instance

        Methods
        -------
        GetUnits()
            (inherited from 'Val') returns 'True' if the BGVal-instance is set to physical units and 'False' if set to numerical units.
        SetUnits()
            (inherited from 'Val') convert the 'value' of the BGVal-instance to physical units or numerical units.
        GetConversion()
            (inherited from 'Val') retrieve the conversion factor for this BGVal-instance
        """

        name=Qname
        u_H0 = H0
        u_MP = MP
        dtype = Qdtype
        def __init__(self, value, BGSystem):
            super().__init__(value, BGSystem)

    return BGVal


def BGFunc(Qname : str, args : list, H0 : int, MP : int, Qdtype : np.dtype=np.float64):
    """
    A class factory used to define dimensionful functions of cosmological quantities with a specific scaling behaviour w.r.t
    an inverse time scale (typically Hubble rate) and a mass scale (typically the Planck mass).

    Parameters
    ----------
    name : str
        the functions name
    args : list of BGVal
        a list of BGVals indicating the arguments of the BGFunc.
    H0 : int
        the functions' scaling w.r.t. a typical inverse time scale
    MP : int
        the functions' scaling w.r.t. a typical mass scale
    dtype : Numpy Data Type
        the data type of the BGVal object.
        
    Returns
    -------
    BGVal
        a class representing a dimensionful function of a cosmological quantity, e.g., a scalar potential.
    """

    if not( np.issubdtype(Qdtype, np.floating)
             or
             np.issubdtype(Qdtype, np.complexfloating)
            ):
        raise TypeError("BGFunc's data-type must be a subtype of np.floating or np.complexfloating.")
    
    class BGFunc(Func):
        """
        This class represents a dimensionful function of cosmological quantities with a given mass dimension and scaling w.r.t.
        the Hubble rate and Planck mass.
        Typical functions are:
            - scalar potential: scales with H^2, M_pl^2, is a function of a scalar-field amplitude
            - scalar--gauge-field coupling: scales with H^0, M_pl^1, is a function of a scalar-field amplitude
        BGFuncs inherit from 'Func' including __call_ and methods related to unit conversion. 

        Attributes
        ----------
        name : str
            the quantities name
        Args : list of BGVal
            a list of BGVals indicating the arguments of the BGFunc.
        u_H0 : int
            the quantities' scaling w.r.t. a typical inverse time scale
        u_MP : int
            the quantities' scaling w.r.t. a typical mass scale
        dtype : Numpy Data Type
            the data type of the BGVal object.

        Methods
        -------
        GetUnits()
            (inherited from 'Func') returns 'True' if the BGVal-instance is set to physical units and 'False' if set to numerical units.
        SetUnits()
            (inherited from 'Func') convert the 'value' of the BGVal-instance to physical units or numerical units.
        GetConversion()
            (inherited from 'Func') retrieve the conversion factor for this BGVal-instance
        """

        name=Qname
        u_H0 = H0
        u_MP = MP
        Args = args
        dtype = Qdtype
        def __init__(self, func, BGSystem):
            super().__init__(func, BGSystem)

    return BGFunc


class BGSystem:
    """
    A collection of cosmological background quantities like cosmic time, Hubble rate, scale-factor etc.
    Instances of this class define two base unit-systems, 'physical units' and 'numerical units' by setting two energy scales:
       - H0: a reference Hubble rate (typically H(t=t_init) such that numerical units correspond to Hubble units at t_init).
       - MP: a reference mass scale in Planck units (typically set to 1. such that physical units are equivalent to Planck units).
    All quantities associated with this BGSystem can be converted from one unit-system to another by use of these energy-scales.
    This class is the fundamental building block of the GEF-code. 

    Attributes
    ----------
    H0 : float
        the Hubble-rate energy scale used for conversion between numerical and physical units
    MP : float
        the Planck-mass energy scale used for conversions between numerical and physical units
    'QuantityName' : Val or Func
        The BGSystem can contain a number of attributes associated to instances of BGVal/BGFunc objects.
        These attributes are assigned using the names of the Val and Func objects (see examples below).

    Methods
    -------
    FromBGSystem()
        initialise a new BGSystem from an existing BGSystem wihtout carying over BGVal and BGFunc instances
    Initialise()
        Initialise a BGVal or BGFunc object with a specific value
    SetUnits()
        change the BGSystem-instance between unit systems
    GetUnits()
        returns "True" if the BGSystem-instance is in physical units, "False" if it is in numerical units
    ObjectSet():
        return a set of all BGVal and BGFunc objects
    ObjectNames()
        return a list of names for all BGVal and BGFunc objects
    ValueList() / FunctionList()
        return a list of all BGVal/BGFunc instances
    ValueNames() / FunctionNames()
        return a list of names for all instantiated BGVal/BGFunc objects
    CreateCopySystem()
        create an exact copy of the current BGSystem including BGVal and BGFunc instances.

    Example
    -------
    ... #define two new variables cooresponding to physical time and Hubble rate.
    >>> time = BGVal("t", H0=-1, MP=0)
    >>> Hubble = BGVal("H", H0=-1, MP=0)
    ...
    ... #Create a BGSystem with the Hubble rate set to 1e-5*Mpl
    >>> U = BGSystem({time, Hubble}, H0, MP)
    ... 
    ... #The BGSystem knows about the new variables time and Hubble
    >>> print(U.ObjectNames()) #prints ["t", "H"]
    ...
    ... #Instantiate the objects "t" and "H" using the units defined by U:
    >>> U.Initialise("t")(1e5)
    >>> U.Initialise("H")(1e-5)
    ...
    ... #The BGSystem now recognises "t" and "H" as keys, you can access them like:
    >>> print(f"In Planck units, the Hubble rate at time {U.t} is {U.H}.") #prints 1e5 and 1e-5
    ...
    ... #Convert everything to numerical units
    >>> U.SetUnits(False)
    >>> print(f"In numerical units, the Hubble rate at time {U.t} is {U.H}.") #prints 1 and 1
    """

    def __init__(self, quantities, H0, MP):
        self.H0 = H0
        self.MP = MP
        self.__units=True
        for quantity in quantities:
            setattr(self, f"_{quantity.name}", quantity)
        
    @classmethod
    def FromBGSystem(cls, system):
        H0 = system.H0
        MP = system.MP
        quantities = system.ObjectSet()
        newinstance = cls(quantities, H0, MP)
        return newinstance

    def Initialise(self, quantity):
        def init(obj):
            q = getattr(self, f"_{quantity}")
            setattr(self, quantity, q(obj, self))
        return init
        
    def SetUnits(self, units):
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Quantity):
                obj.SetUnits(units)
        self.__units=units
        return

    def GetUnits(self):
        return self.__units
    
    def ObjectSet(self):
        objects = []
        for var in vars(self):
            obj = getattr(self, var)
            if inspect.isclass(obj):
                if issubclass(obj, Quantity):
                    objects.append(obj)      
        return set(objects)
    
    def ObjectNames(self):
        names = []
        for obj in self.ObjectSet():
            names.append(obj.name)
        return names
    
    def ValueList(self):
        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Val):
                vals.append(obj)    
        return vals
    
    def ValueNames(self):
        names = []
        for val in self.ValueList():
            names.append(val.name)
        return names

    def FunctionList(self):
        funcs = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Func):
                    funcs.append(obj)      
        return funcs
    
    def FunctionNames(self):
        names = []
        for val in self.FunctionList():
            names.append(val.name)
        return names

    def CreateCopySystem(self):
        units = self.GetUnits()
        newsystem = BGSystem.FromBGSystem(self)
        self.SetUnits(True)

        values = self.ValueList()
        funcs = self.FunctionList()
        
        for value in values:
            obj = deepcopy(value.value)
            newsystem.Initialise(value.name)(obj)

        for func in funcs:
            obj = func.GetBaseFunc()
            newsystem.Initialise(func.name)(obj)
        
        self.SetUnits(units)
        
        return newsystem

    def Remove(self, name):
        delattr(self, name)
        delattr(self, f"_{name}")
        return
    
    def AddBGVal(self, name, H0units, MPunits):
        setattr(self, f"_{name}",
                 BGVal(name, H0units, MPunits))
        return
    
    def AddBGFunc(self, name, args, H0units, MPunits):
        setattr(self, f"_{name}",
                 BGFunc(name, args, H0units, MPunits))
        return
    
    def AddValue(self, name, value, H0units, MPunits):
        self.AddBGVal(name, H0units, MPunits)
        self.Initialise(name)(value)
        return
    
    def AddFunction(self, name, args, function, H0units, MPunits):
        self.AddBGFunc(name, args, H0units, MPunits)
        self.Initialise(name)(function)
        return
    
class Quantity:
    """
    The parent class for BGVal and BGFunc.
    This represents a cosmological quantity with a given mass dimension and scaling w.r.t. the Hubble rate and Planck mass.
    Typical quantities are:
        - time: scales with H^(-1), M_pl^0
        - time-derivative: scales with H^1, M_pl^0
        - scalar-field amplitude: scales with H^0, M_pl^1
        - scalar-field potential: scales with H^2 M_pl^2
        - gauge-fields: scales with H^1, M_pl^0 (i.e. like a derivative)
    
    Attributes
    ----------
    name : str
        the quantities name
    u_H0 : int
        an integer indicating how the Quantity scales with Hubble rate
    u_MP : int
        an integer indicating how the Quantity scales with the Planck mass
    dtype : type
        the data type of the Quantity object.
    """

    name= ""
    u_H0 = 0
    u_MP = 0
    dtype = np.float64

    def __repr__(self):
        return f"{self.name}(H0={self.u_H0}, MP={self.u_MP})"

class Val(Quantity):
    """
    The parent class for BGVal. Determines the base arithmetic operations of a BGVal and how
    conversions between physical units and numerical units are handled.
    An instance of this object can be used like a 1-D Numpy-Array for mathematical operations and iterating.

    Attributes
    ----------
    value : NDArray
        a 1-dimensional array of values set to the current units of this BGVal-instance.
        This attribute is used for all arithmetic operations and is converted to physical and numerical units via SetUnits()
    massdim : int
        the mass dimension of this Val-instance
    
    Methods
    -------
    GetUnits()
        returns 'True' if the Val-instance is set to physical units and 'False' if set to numerical units.
    SetUnits()
        convert the 'value' of the Val-instance to physical units or numerical units.
    GetConversion()
        retrieve the conversion factor for this Val-instance

    """
    def __init__(self, value : ArrayLike, BGSystem : BGSystem):
        self.__DefineMassdim__()

        self.value = np.asarray(value, dtype=self.dtype)
        self.__units = BGSystem.GetUnits()
        
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    @classmethod
    def __DefineMassdim__(cls):
        cls.massdim = cls.u_H0+cls.u_MP
        return

    def __str__(self):
        if self.__units==False:
            return f"{self.name} (Unitless): {self.value}"
        elif self.__units==True:
            return f"{self.name} (Unitful): {self.value}"
        
    def GetUnits(self):
        return self.__units
    
    def SetUnits(self, units):
        if isinstance(self.value, type(None)):
            self.__units=units
            return
        if units:
            self.__Unitful()
        elif not(units):
            self.__Unitless()
        return

    def __Unitless(self):
        if self.__units:
            self.value /= self.__Conversion
        self.__units=False
        return
    
    def __Unitful(self):
        if not(self.__units):
            self.value *= self.__Conversion
        self.__units=True
        return
    
    def SetValue(self, value):
        #set the value of the array element assuming the units of the BG system
        self.value = np.asarray(value)
        return
    
    def GetConversion(self):
        return self.__Conversion
    
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
    
class Func(Quantity):
    """
    The parent class for BGFunc. Determines the __call__ method of a BGFunc and how
    conversions between physical units and numerical units are handled.
    An instance of this object can be used like a Callable of BGVals or arithmetic-type objects using the 
    call signature indicated by "Args" (see below).)
    
    Methods
    -------
    __call__()
        returns the result of the underlying basefunction f(*args) according to f( args (in physical units) ).
        The result of this operation are returned in numerical / physical units depending on the units of the current Func-instance.
        If called by a BGVal, the conversion of the argument is handled by knowledge of the current BGVal-instance's 
        
    GetBaseFunc()
        returns the underlying function defining the Func-instance.
    GetUnits()
        returns 'True' if the Func-instance is set to physical units and 'False' if set to numerical units.
    SetUnits()
        convert the 'value' of the Func-instance to physical units or numerical units.
    GetConversion()
        retrieve the conversion factor for this Func-instance

    """
    Args = []
    def __init__(self, func, BGSystem):
        super().__init__()
        func = np.vectorize(func, otypes=[self.dtype])
        
        try:
            testargs = [1.0 for arg in self.Args]
            assert func(*testargs).dtype
        except TypeError:
            raise TypeError("The number of non-default arguments of 'func' needs to match 'len(self.Args)'.")
        except ValueError:
            raise ValueError("'func' must return a single value which can be converted to '{self.dtype}'")
        
        self.__basefunc = func

        self.__units = BGSystem.GetUnits()
        self.__ArgConversions = [(BGSystem.H0**arg.u_H0*BGSystem.MP**arg.u_MP)
                                 for arg in self.Args]
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    def __call__(self, *args):
        units = self.GetUnits()
        def floathandler(x, i):
            return x*self.__ArgConversions[i]**(1-units)
        
        def Valhandler(x, i):
            conv = x.GetConversion()
            assert self.__ArgConversions[i] == conv
            pow = (1 - x.GetUnits())
            return x*conv**pow

        typedic = {Val : Valhandler}

        args = [typedic.get(arg.__class__.__bases__[0], floathandler)(arg, i) for i, arg in enumerate(args)]

        return self.__basefunc(*args)/self.__Conversion**(1-units)
        
    def GetUnits(self):
        return self.__units
    
    def SetUnits(self, units):
        if units:
            self.__Unitful()
        elif not(units):
            self.__Unitless()
        return
    
    def __Unitless(self):
        self.__units=False
        return
    
    def __Unitful(self):
        self.__units=True
        return
        
    def GetBaseFunc(self):
        return self.__basefunc
    
    def GetConversion(self):
        return self.__Conversion
    
    def GetArgConversions(self):
        return self.__ArgConversions

class IncompatibleQuantitiesException(Exception):
    pass
