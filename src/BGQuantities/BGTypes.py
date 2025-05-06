import numpy as np
from copy import deepcopy
import inspect

class Quantity:
    name= ""
    u_H0 = 0
    u_MP = 0

    def __repr__(self):
        return f"{self.name}(H0={self.u_H0}, MP={self.u_MP})"

class BGVal(Quantity):
    def __init__(self, value, BGSystem):
        super().__init__()
        self.value = value
        self.__units = BGSystem.GetUnits()
        self.massdim = self.u_H0+self.u_MP
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    def __str__(self):
        if self.__units==None:
            return f"{self.name}: {self.value}"
        if self.__units==False:
            return f"{self.name} (Unitless): {self.value}"
        if self.__units==True:
            return f"{self.name} (Unitful): {self.value}"
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __len__(self):
        return len(self.value)
    
    def __round__(self):
        return round(self.value)
    
    def __abs__(self):
        return abs(self.value)
    
    def __neg__(self):
        return -self.value
    
    def __pos__(self):
        return +self.value
    
    def __add__(self, other):
        return self.value + other
        
    __radd__ = __add__
    
    def __sub__(self, other):
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
        return self.value == other
            
    def __ne__(self, other):
        return self.value != other
    
    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return  self.value <= other

    def __ge__(self, other):
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
    
class BGFunc(Quantity):
    def __init__(self, func, BGSystem):
        super().__init__()
        self.__basefunc = func
        self.__Conversion = (BGSystem.H0**self.u_H0*BGSystem.MP**self.u_MP)

    def __call__(self, value):
        if isinstance(value, BGVal):
            valueconversion = value.GetConversion()
            pow = (1 - value.GetUnits())
            return self.__basefunc(value*valueconversion**pow)/self.__Conversion**pow 
        
    def GetBaseFunc(self):
        return self.__basefunc
    
    def GetConversion(self):
        return self.__Conversion

def DefineQuantity(Qname, H0, MP, isfunc=False):
    if not(isfunc):
        class Val(BGVal):
            name=Qname
            u_H0 = H0
            u_MP = MP
            def __init__(self, value, BGSystem):
                super().__init__(value, BGSystem)

        return Val
    
    elif isfunc:
        class Func(BGFunc):
            name=Qname
            u_H0 = H0
            u_MP = MP
            def __init__(self, func, BGSystem):
                super().__init__(func, BGSystem)

        return Func

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
    
    def AddQuantity(self, name, H0units, MPunits, isfunc=False):
        setattr(self, f"_{name}",
                 DefineQuantity(name, H0units, MPunits, isfunc=isfunc))
    
    def AddValue(self, name, value, H0units, MPunits):
        self.AddQuantity(name, H0units, MPunits, isfunc=False)
        self.Initialise(name)(value)
        return
    
    def AddFunction(self, name, function, H0units, MPunits):
        self.AddQuantity(name, H0units, MPunits, isfunc=True)
        self.Initialise(name)(function)
        return
        
    def SetUnits(self, units):
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, BGVal):
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
    
    def ValList(self):
        vals = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, BGVal):
                vals.append(obj)    
        return vals

    def FuncList(self):
        funcs = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, BGFunc):
                    funcs.append(obj)      
        return set(funcs)
    
    def ObjectNames(self):
        names = []
        for obj in self.ObjectSet():
            names.append(obj.name)
        return names

    def ValueNames(self):
        names = []
        for val in self.ValList():
            names.append(val.name)
        return names

    def FunctionNames(self):
        names = []
        for val in self.FuncList():
            names.append(val.name)
        return names

    def CreateCopySystem(self):
        units = self.GetUnits()
        newsystem = BGSystem.InitialiseFromBGSystem(self)
        self.SetUnits(True)

        values = self.ValList()
        funcs = self.FuncList()
        
        for value in values:
            obj = deepcopy(value.value)
            newsystem.Initialise(value.name)(obj)

        for func in funcs:
            obj = func.GetBaseFunc()
            newsystem.Initialise(func.name)(obj)
        
        self.SetUnits(units)
        
        return newsystem
        



