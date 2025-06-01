import numpy as np
from copy import deepcopy
import inspect

from numpy.typing import ArrayLike, NDArray
from typing import Callable

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
        Initialise a new BGSystem from an existing BGSystem instance.
    Initialise()
        Initialise a BGVal / BGFunc instance associated with this BGSystem instance.
    SetUnits()
        Convert the BGSystem and all its BGVal / BGFunc instances betweem numerical and physical units.
    GetUnits()
        Get a boolean representing the current unit system of this BGSystem instance.
    ObjectSet():
        Get a set of all BGVal / BGFunc objects (not necessarily instantiated) attributed to this BGSystem.
    ObjectNames()
        Get a list of names for all BGVal / BGFunc objects (not necessarily instantiated) attributed to this BGSystem.
    ValueList() / FunctionList()
        Get a list of all BGVal / BGFunc instances attributed to this BGSystem.
    ValueNames() / FunctionNames()
        Get a list of names for all BGVal / BGFunc instances attributed to this BGSystem.
    CreateCopySystem()
        Create a copy of tbis BGSystem including all instances of BGVal's and BGFunc's

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
    def FromBGSystem(cls, system : 'BGSystem') -> 'BGSystem':
        """
        Initialise a new BGSystem from an existing BGSystem instance

        Parameters
        ----------
        system : BGSystem
            the BGSystem instance used for initialising the new BGSystem

        Returns
        -------
        BGSystem
            the new BGSystem instance
        """

        H0 = system.H0
        MP = system.MP
        quantities = system.ObjectSet()
        newinstance = cls(quantities, H0, MP)
        return newinstance

    def Initialise(self, quantity : str) -> Callable:
        """
        Initialise a BGVal / BGFunc instance associated with this BGSystem instance

        Parameters
        ----------
        quantity : str
            the 'name' attribute of the BGVal / BGFunc which is to be instantiated.

        Returns
        -------
        Callable
            a function used to initialise the BGVal / BGFunc with a arithemetic type / Callable
        """

        def init(obj):
            """
            Initialise a BGVal / BGFunc instance with an arithmetic type / Callable.
            This adds the BGVal / BGFunc instance as a new attribute for the current BGSystem, with the attribute name
            corresponding to the BGVal / BGFunc instance's 'name' attribute. 

            Parameters
            ----------
            obj : NDArray or Callable
                the NDArray / Callable with which the BGVal / BGFunc is to be instantiated.
            """

            q = getattr(self, f"_{quantity}")
            setattr(self, quantity, q(obj, self))
        return init
        
    def SetUnits(self, units : bool):
        """
        Convert the BGSystem and all its BGVal / BGFunc instances betweem numerical and physical units.

        Parameters
        ----------
        units : bool
            If 'True', switch to physical units, if 'False', switch to numerical units.
        """

        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Quantity):
                obj.SetUnits(units)
        self.__units=units
        return

    def GetUnits(self) -> bool:
        """
        Get a boolean representing the current unit system of this BGSystem instance.

        Returns
        -------
        bool
            'True' if the system is in physical units, 'False' if in numerical units.
        """

        return self.__units
    
    def ObjectSet(self) -> set[object]:
        """
        Get a set of all BGVal / BGFunc objects (not necessarily instantiated) attributed to this BGSystem.

        Returns
        -------
        set
            the set of associated BGVal / BGFunc objects.
        """

        objects = []
        for var in vars(self):
            obj = getattr(self, var)
            if inspect.isclass(obj):
                if issubclass(obj, Quantity):
                    objects.append(obj)      
        return set(objects)
    
    def ObjectNames(self) -> list[str]:
        """
        Get a list of names for all BGVal / BGFunc objects (not necessarily instantiated) attributed to this BGSystem.

        Returns
        -------
        list of str
            the list of names.
        """

        names = []
        for obj in self.ObjectSet():
            names.append(obj.name)
        return names
    
    def ValueList(self) -> list['Val']:
        """
        Get a list of all BGFunc instances attributed to this BGSystem.

        Returns
        -------
        list of BGVal
            the list of BGVal instances.
        """

        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Val):
                vals.append(obj)    
        return vals
    
    def ValueNames(self) -> list[str]:
        """
        Get a list of names for all BGVal instances attributed to this BGSystem.

        Returns
        -------
        list of str
            the list of names.
        """

        names = []
        for val in self.ValueList():
            names.append(val.name)
        return names

    def FunctionList(self) -> list['Func']:
        """
        Get a list of all BGFunc instances attributed to this BGSystem.

        Returns
        -------
        list of BGFunc
            the list of BGFunc instances.
        """

        funcs = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Func):
                    funcs.append(obj)      
        return funcs
    
    def FunctionNames(self) -> list[str]:
        """
        Get a list of names for all BGFunc instances attributed to this BGSystem.

        Returns
        -------
        list of str
            the list of names.
        """

        names = []
        for val in self.FunctionList():
            names.append(val.name)
        return names

    def CreateCopySystem(self) -> 'BGSystem':
        """
        Create a copy of tbis BGSystem including all instances of BGVal's and BGFunc's.

        Returns
        -------
        BGSystem
            a copy of this BGSystem
        """

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

    def Remove(self, name : str):
        """
        Remove a BGVal / BGFunc object including it's instance from the BGSystem.

        Parameters
        ----------
        name : str
            the name of the BGVal / BGFunc object
        """

        delattr(self, name)
        delattr(self, f"_{name}")
        return
    
    def AddBGVal(self, name : str, H0units : int, MPunits : int):
        """
        Add a BGVal object to the BGSystem.

        Parameters
        ----------
        name : str
            the name of the BGVal object.
        H0units : int
            the BGVal's 'u_H0' parameter.
        MPunits : int
            the BGVal's 'u_MP' parameter.
        """

        setattr(self, f"_{name}",
                 BGVal(name, H0units, MPunits))
        return
    
    def AddBGFunc(self, name : str, args : list['Val'], H0units : int, MPunits : int):
        """
        Add a BGFunc object to the BGSystem.

        Parameters
        ----------
        name : str
            the name of the BGVal object.
        args : list of BGVal
            the BGFunc's 'Args' parameter.
        H0units : int
            the BGFunc's 'u_H0' parameter.
        MPunits : int
            the BGFunc's 'u_MP' parameter.
        """

        setattr(self, f"_{name}",
                 BGFunc(name, args, H0units, MPunits))
        return
    
    def AddValue(self, name : str, value : NDArray, H0units : int, MPunits : int):
        """
        Add a BGVal object to the BGSystem and instantiate it.

        Parameters
        ----------
        name : str
            the name of the BGVal object.
        value : NDArray or float
            the value with which to instantiate the BGVal object.
        H0units : int
            the BGVal's 'u_H0' parameter.
        MPunits : int
            the BGVal's 'u_MP' parameter.
        """

        self.AddBGVal(name, H0units, MPunits)
        self.Initialise(name)(value)
        return
    
    def AddFunction(self, name : str, args : list['Val'], function : Callable, H0units : int, MPunits : int):
        """
        Add a BGFunc object to the BGSystem and instantiate it.

        Parameters
        ----------
        name : str
            the name of the BGVal object.
        args : list of BGVal
            the BGFunc's 'Args' parameter.
        function : Callable
            the function with which to instantiate the BGFunc object.
        H0units : int
            the BGFunc's 'u_H0' parameter.
        MPunits : int
            the BGFunc's 'u_MP' parameter.
        """

        self.AddBGFunc(name, args, H0units, MPunits)
        self.Initialise(name)(function)
        return
    
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

        Example 1 (defining a BGVal) 
        -----------------------------
        ... #create a BGVal object: the electric field expectation value E^2
        >>> E0 = BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 
        ... 
        ... #Add 'E0' to a BGSystem in Planck units (M_pl = 1) and H0 = 1e-5*M_pl:
        >>> U = BGSystem([E0], H0=1e-5, MP=1.)
        ... 
        ... #initialise E0 with some value in Planck units:
        >>> U.Initialise("E0")( 6e-10 )
        ... #U.E0 is now an instance of BGVal

        Example 2 (converting the units of a BGVal)
        -------------------------------------------
        ... #Calling in Planck units is straight forwards:
        >>> print( U.E0.value )#gives 6e-10 (in Planck units)
        ... 
        ... #Switch E0 to numerical units changes the value stored in U.E0:
        >>> U.E0.SetUnits(False)
        >>> print( U.E0.value )#gives 6e10 = 6e-10 * U.H0**4 (in numerical units)
        """

        name=Qname
        u_H0 = H0
        u_MP = MP
        dtype = Qdtype
        def __init__(self, value, BGSystem):
            super().__init__(value, BGSystem)

    return BGVal


def BGFunc(Qname : str, args : list['Val'], H0 : int, MP : int, Qdtype : np.dtype=np.float64):
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
    Qdtype : Numpy data type returned by the BGFunc
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

        A call to a BGFunc (inherited from 'Func') returns the result of the underlying basefunction, f(*args), according to f( *args (in physical units) ).
        The result of this operation are returned in numerical / physical units depending on the units of the current BGFunc-instance.
        If called by a BGVal, the conversion of the argument is done using the units of the BGVal-instance. 
        If called by a non-BGVal arithmetic data type, it is assumed that the argument is in the same unit system as the BGFunc instance. 

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
        GetBaseFunc()
            (inherited from 'Func') returns the underlying function defining the BGFunc-instance.
        GetUnits()
            (inherited from 'Func') returns 'True' if the BGFunc-instance is set to physical units and 'False' if set to numerical units.
        SetUnits()
            (inherited from 'Func') change __call__() of the BGFunc-instance to physical units or numerical units.
        GetConversion()
            (inherited from 'Func') retrieve the conversion factor for this BGFunc-instance
        
        Example 1 (defining a BGFunc) 
        -----------------------------
        ... #create a BGVal object: the electric field expectation value E^2
        >>> E0 = BGVal("E0", H0=4, MP=0) #since A_mu scales like d / dx^mu 
        ...
        ... #define a BGFunc object: rhoE, the electric field energy density
        >>> rhoE = BGFunc("rhoE", args=[E0], H0=2, MP=2) # since 3 * M_pl^2 * H^2 = rho
        ... 
        ... #collect both in a BGSystem in Planck units (M_pl = 1) and H0 = 1e-5*M_pl:
        >>> U = BGSystem([E0, rhoE], H0=1e-5, MP=1.)
        ... 
        ... #initialise E0 with some value in Planck units:
        >>> U.Initialise("E0")( 6e-10 )
        ... #define a function: rhoE = 0.5*E^2
        >>> func = lambda x: 0.5 * x
        >>> U.Initialise("rhoE")( func )
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
        >>> U.E0.SetUnits(False) #E0.value = 6e10
        >>> print( U.rhoE(U.E0) ) #gives 3e-10 (still in Planck units)
        ... 
        ... #Switching U.rhoE to numerical units means calling rhoE returns values in numerical units:
        >>> U.rhoE.SetUnits(False)
        >>> print( U.rhoE(U.E0) ) #gives 3. = 3e-10 / (U.H0*U.MP)**2 (in numerical units)
        ... 
        ... #Again, this outcome does not depend on the units of E0:
        >>> U.E0.SetUnits(True)
        >>> print( U.rhoE(U.E0) ) #gives 3. = 3e-10 / (U.H0*U.MP)**2 (in numerical units)

        Example 3 (calling a BGFunc by a float)
        ---------------------------------------
        ... #instead of calling rhoE by E0, we can call it by a float:
        >>> val = 6e-10
        ... 
        ... #First the behaviour if rhoE is in physical units:
        >>> U.rhoE.SetUnits(True)
        >>> print( U.rhoE( val ) )  #gives 3e-10 (in Planck units)
        ... 
        ... #Things are different if rhoE is in numerical units:
        >>> U.rhoE.SetUnits(False)
        >>> print( U.rhoE(val) ) #gives 3e-20 = 0.5* (6e-10*U.H0**4) / (U.H0*U.MP)**2
        ... 
        ... #since val does not have units, it is assumed to be in the units of rhoE.
        ... # If you want this to give the correct result, you would need to convert val by hand:
        >>> print( U.rhoE(val/U.H0**4) ) #gives 3., the expected result in numerical units. 

        """

        name=Qname
        u_H0 = H0
        u_MP = MP
        Args = args
        dtype = Qdtype
        def __init__(self, func, BGSystem):
            super().__init__(func, BGSystem)

    return BGFunc
    
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
        A boolean corresponding to the current units of the Val instance.
    SetUnits()
        Convert the Val instnace betweem numerical and physical units.
    GetConversion()
        Get the conversion factor between numerical and physical units for this Val instance.
    """

    def __init__(self, value : NDArray, BGSystem : BGSystem):
        self.__DefineMassdim__()

        self.value = np.asarray(value, dtype=self.dtype)
        self.__units = BGSystem.GetUnits()
        
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    @classmethod
    def __DefineMassdim__(cls):
        """
        Set the 'massdim' class attribute.
        """

        cls.massdim = cls.u_H0+cls.u_MP
        return

    def __str__(self) -> str:
        """
        The class represented as a string

        Returns
        -------
        str
            the string representation.
        """

        if self.__units==False:
            return f"{self.name} (numerical): {self.value}"
        elif self.__units==True:
            return f"{self.name} (physical): {self.value}"
        
    def GetUnits(self):
        """
        A boolean corresponding to the current units of the Val instance.

        Returns
        -------
        bool
            'True' if the value is in physical units, 'False' if in numerical units.
        """

        return self.__units
    
    def SetUnits(self, units : bool):
        """
        Convert the Val instnace betweem numerical and physical units.

        Parameters
        ----------
        units : bool
            If 'True', switch to physical units, if 'False', switch to numerical units.
        """

        if isinstance(self.value, type(None)):
            self.__units=units
            return
        if units and not(self.__units):
            self.value *= self.__Conversion
        elif not(units) and self.__units:
            self.value /= self.__Conversion
        self.__units=units
        return
    
    def SetValue(self, value : NDArray):
        """
        Overwrite the 'value' attribute.

        Parameters
        ----------
        value : NDarray or float
            the new value.
        """

        self.value = np.asarray(value)
        return
    
    def GetConversion(self) -> float:
        """
        Get the conversion factor between numerical and physical units for this Val instance.

        Returns
        -------
        float
            the conversion factor
        """

        return self.__Conversion
    
    #The following methods ensure that a 'Val' instance can be used as an array concerning mathematical operations and indexing.
    
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
    call signature indicated by "Args".

    A call to a Func returns the result of the underlying basefunction f(*args) according to f( *args (in physical units) ).
    The result of this operation are returned in numerical / physical units depending on the units of the current Func-instance.
    If called by a BGVal, the conversion of the argument is done using the units of the BGVal-instance. 
    If called by a non-BGVal arithmetic data type, it is assumed that the argument is in the same unit system as the BGFunc instance.
    
    Methods
    -------
    GetBaseFunc()
        Get the underlying function defining the __call__ method.
    GetUnits()
        A boolean corresponding to the current units of the Func instance.
    SetUnits()
        Convert the Func betweem numerical and physical units.
    GetConversion()
        Get the conversion factor between numerical and physical units for this Func instance.

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
        
    def GetUnits(self) -> bool:
        """
        A boolean corresponding to the current units of the Func instance.

        Returns
        -------
        bool
            'True' if the value is in physical units, 'False' if in numerical units.
        """

        return self.__units
    
    def SetUnits(self, units : bool):
        """
        Convert the Func betweem numerical and physical units.

        Parameters
        ----------
        units : bool
            If 'True', switch to physical units, if 'False', switch to numerical units.
        """

        self.__units = units
        return
        
    def GetBaseFunc(self) -> Callable:
        """
        Get the underlying function defining the __call__ method.

        Returns
        -------
        Callable:
            the function.
        """
        return self.__basefunc
    
    def GetConversion(self) -> float:
        """
        Get the conversion factor between numerical and physical units for this Func instance.

        Returns
        -------
        float
            the conversion factor
        """

        return self.__Conversion
    
    def GetArgConversions(self) -> list[float]:
        """
        Get a list of conversion factors between numerical and physical units for each argument.

        Returns
        -------
        list of float
            the list of conversion factors
        """

        return self.__ArgConversions
    
    #the dunder method defines the call signature as described in the class documentation
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
