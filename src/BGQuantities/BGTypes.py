import numpy as np
    
class BGSystem:
    def __init__(self, values, functions, H0, MP):
        self.H0 = H0
        self.MP = MP
        for key, item in values.items():
            #initialise value in dictionary with given name and unit conversion. Value is assumed Unitful
            self.AddValue(key, item["value"], item["H0"], item["MP"])
        for key, item in functions.items():
            self.AddFunction(key, item["func"], item["H0"], item["MP"])
        self.__units=True

    def AddValue(self, name, value, H0units, MPunits, units=True):
        setattr(self, name, self.BGVal(name, value, H0units, MPunits, self.H0, self.MP, units=units))
        return
    
    def RemoveValue(self, name):
        delattr(self, name)
        return
    
    def AddFunction(self, name, function, H0units, MPunits):
        setattr(self, name, self.BGFunc(name, function, H0units, MPunits, self.H0, self.MP))
        
    def RemoveFunction(self, name):
        delattr(self, name)
        return
        
    def SetUnits(self, units):
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, self.BGVal):
                obj.SetUnits(units)
        self.__units=units
        return

    def GetUnits(self):
        return self.__units

    def ListValues(self):
        valuelist = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, self.BGVal):
                valuelist.append(obj.name)
        return valuelist

    def ListFunctions(self):
        funclist = []
        for var in vars(self):
            obj = getattr(self, var)
            if isinstance(obj, self.BGFunc):
                funclist.append(obj.name)
        return funclist
    
    def CopySystem(self):
        values = self.ListValues()
        funcs = self.ListFunctions()
        valuedic = dict(zip(values, [{"value":None, "H0":0, "MP":0} for v in values]))
        for value in values:
            obj = getattr(self, value)
            valuedic[value]["value"] = obj.value
            valuedic[value]["H0"] = obj.u_H0
            valuedic[value]["MP"] = obj.u_MP

        funcdic = dict(zip(funcs, [{"func":None, "H0":0, "MP":0} for f in funcs]))
        for func in funcs:
            obj = getattr(self, func)
            funcdic[func]["func"] = obj.GetBaseFunc()
            funcdic[func]["H0"] = obj.u_H0
            funcdic[func]["MP"] = obj.u_MP
        
        return BGSystem(valuedic, funcdic, self.H0, self.MP)
    
    class BGVal:
        def __init__(self, name, value, H0units, MPunits, H0, MP=1, units=True):
            self.value = value
            self.name = name
            self.__units = True
            self.u_H0 = H0units
            self.u_MP = MPunits
            self.massdim = self.u_H0+self.u_MP
            self.__Conversion = (H0**H0units*MP**MPunits)

        def __str__(self):
            if self.__units==None:
                return f"{self.name}: {self.value}"
            if self.__units==False:
                return f"{self.name} (Unitless): {self.value}"
            if self.__units==True:
                return f"{self.name} (Unitful): {self.value}"
        
        def __repr__(self):
            return f"BGVal(name='{self.name}', H0Units={self.u_H0}, MPunits={self.u_MP})"
        
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
            """if isinstance(other, BGVal):
                assert (self.massdim==other.massdim)
                #always add quantities with units
                return (self.value*self.__Conversion**(1-self.__units) 
                        + other.value*other.GetConversion()**(1-other.GetUnits()))"""
            return self.value + other
            
        __radd__ = __add__
        
        def __sub__(self, other):
            """if isinstance(other, BGVal):
                assert (self.massdim==other.massdim)
                #always add quantities with units
                return (self.value*self.__Conversion**(1-self.__units) 
                        - other.value*other.GetConversion()**(1-other.GetUnits()))"""
            return self.value - other
            
        __rsub__ = __sub__

        def __mul__(self, other):
            return self.value * other
            """if isinstance(other, BGVal):
                return self.value * other.value
            else:
                return self.value * other"""
            
        __rmul__ = __mul__
        
        def __floordiv__(self, other):
            return self.value // other
            
        def __rfloordiv__(self, other):
            return other // self.value
        
        def __truediv__(self, other):
            return self.value / other
            """if isinstance(other, BGVal):
                return self.value / other.value
            else:
                return self.value / other"""
            
        def __rtruediv__(self, other):
            return other / self.value
            """if isinstance(other, BGVal):
                return other.value / self.value
            else:
                return other / self.value"""
        
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
        
        def SetVal(self, value, units=True):
            self.value = np.asarray(value)
            self.__units = units

        def GetConversion(self):
            return self.__Conversion
    

    class BGFunc:
        def __init__(self, name, func, H0units, MPunits, H0, MP):
            self.u_H0 = H0units
            self.u_MP = MPunits
            self.__basefunc = func
            self.name = name
            self.__Conversion = (H0**H0units*MP**MPunits)

        def __call__(self, value):
            if isinstance(value, BGSystem.BGVal):
                valueconversion = value.GetConversion()
                pow = (1 - value.GetUnits())
                return self.__basefunc(value.value*valueconversion**pow)/self.__Conversion**pow 
            
        def GetBaseFunc(self):
            return self.__basefunc
        
        def GetConversion(self):
            return self.__Conversion







    




