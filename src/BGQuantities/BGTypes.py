import numpy as np

class BGVal:
    def __init__(self, name, H0units, MPunits, H0, MP=1, units=True):
        self.value = None
        self.name = name
        self.__units = True
        self.u_H0 = H0units
        self.u_MP = MPunits
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
    
    def GetUnits(self):
        return self.__units
    
    def SetUnits(self, units):
        
        if self.value==None:
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
        if isinstance(value, BGVal):
            if value.GetUnits():
                return self.__basefunc(value.value)
            elif not(value.GetUnits()):
                return self.__basefunc(value.value*value.GetConversion())/self.__Conversion
        else:
            raise TypeError
    


    




