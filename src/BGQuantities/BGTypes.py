import numpy as np
from copy import deepcopy
import inspect

class IncompatibleQuantitiesException(Exception):
    pass

class Quantity:
    name= ""
    u_H0 = 0
    u_MP = 0
    dtype = np.float64

    def __repr__(self):
        return f"{self.name}(H0={self.u_H0}, MP={self.u_MP})"

class Val(Quantity):
    def __init__(self, value, BGSystem):
        super().__init__()
        self.value = np.asarray(value, dtype=self.dtype)
        self.__units = BGSystem.GetUnits()
        self.massdim = self.u_H0+self.u_MP
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    def __str__(self):
        if self.__units==False:
            return f"{self.name} (Unitless): {self.value}"
        elif self.__units==True:
            return f"{self.name} (Unitful): {self.value}"
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __len__(self):
        return len(self.value)
    
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
        """if isinstance(other, Val):
            if other.massdim!=0:
                raise IncompatibleQuantitiesException("Cannot exponentiate BGVal with BGVal of non-zero massd imension.")"""
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
    
"""    def __Compatible(self, other, op):
        if isinstance(other, Val):
            if not(self.massdim==other.massdim):
                raise IncompatibleQuantitiesException(
                    f"{op} between BGVal's of mass-dimension {self.massdim} and {other.massdim} is not defined."
                    )
            if not(self.GetUnits() == other.GetUnits()):
                raise IncompatibleQuantitiesException(
                    f"{op} between BGVal's in different units is not defined."
                    )
        return"""

def BGVal(Qname, H0, MP, Qdtype=np.float64):
    if not( np.issubdtype(Qdtype, np.floating)
             or
             np.issubdtype(Qdtype, np.complexfloating)
            ):
        raise TypeError("BGVal's data-type must be a subtype of np.floating or np.complexfloating.")

    class BGVal(Val):
        name=Qname
        u_H0 = H0
        u_MP = MP
        dtype = Qdtype
        def __init__(self, value, BGSystem):
            super().__init__(value, BGSystem)

    return BGVal
    
#Needs to be tested!!!!
class Func(Quantity):
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
            #assert self.__ArgConversions[i] == conv
            pow = (1 - x.GetUnits())
            return x*conv**pow

        typedic = {Val : Valhandler}

        args = [typedic.get(arg.__class__.__bases__[0], floathandler)(arg, i) for i, arg in enumerate(args)]

        return self.__basefunc(*args)/self.__Conversion**(1-units)

        """for i, arg in enumerate(args):
            if isinstance(arg, Val):
                #assert self.__ArgConversions[i] == arg.GetConversion()
                pow = (1 - arg.GetUnits())
                arglist.append(arg.value*arg.GetConversion()**pow)
            else:
                arglist.append(arg*self.__ArgConversions[i]**(1-units))
        
        return self.__basefunc(*arglist)/self.__Conversion**(1-units)"""
        
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


def BGFunc(Qname, args, H0, MP, Qdtype=np.float64):
    if not( np.issubdtype(Qdtype, np.floating)
             or
             np.issubdtype(Qdtype, np.complexfloating)
            ):
        raise TypeError("BGFunc's data-type must be a subtype of np.floating or np.complexfloating.")
    
    class BGFunc(Func):
        name=Qname
        u_H0 = H0
        u_MP = MP
        Args = args
        dtype = Qdtype
        def __init__(self, func, BGSystem):
            super().__init__(func, BGSystem)

    return BGFunc

class BGSystem:
    def __init__(self, quantities, H0, MP):
        self.H0 = H0
        self.MP = MP
        self.__units=True
        for quantity in quantities:
            setattr(self, f"_{quantity.name}", quantity)
        
    @classmethod
    def InitialiseFromBGSystem(cls, system):
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
    
    def ValueList(self):
        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Val):
                vals.append(obj)    
        return vals

    def FunctionList(self):
        funcs = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, Func):
                    funcs.append(obj)      
        return funcs
    
    def ObjectNames(self):
        names = []
        for obj in self.ObjectSet():
            names.append(obj.name)
        return names

    def ValueNames(self):
        names = []
        for val in self.ValueList():
            names.append(val.name)
        return names

    def FunctionNames(self):
        names = []
        for val in self.FunctionList():
            names.append(val.name)
        return names

    def CreateCopySystem(self):
        units = self.GetUnits()
        newsystem = BGSystem.InitialiseFromBGSystem(self)
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
        



