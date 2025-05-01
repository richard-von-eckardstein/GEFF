from src.BGQuantities.BGTypes import BGSystem

class Evolver:
    def __init__(self, quantities):
        y = []
        for quantity in quantities:
            if isinstance(quantity, Evolved):
                pass

    def EvolveQuantities(self):
        pass


class Quantity:
    def __init__(self, name, H0units, MPunits):
        self.name = name
        self.__H0units = H0units
        self.__MPunits = MPunits

    def AddToUnitSystem(self, value, UnitSystem):
        UnitSystem.AddValue(self.name, value, self.__H0units, self.__MPunits)
        return
        
class DerivedAuxiliary(Quantity):
    def __init__(self, name, H0units, MPunits, func):
        super().__init__(name, H0units, MPunits)
        self.func = func
    
class Evolved(Quantity):
    def __init__(self, name, H0units, MPunits, EoM):
        super().__init__(name, H0units, MPunits)
        self.EoM = EoM
        
class ScalarField:
    def __init__(self, name, EoM):
        self.field = Quantity(self, name, 0, 1)
        self.dfield = Quantity(self, f"d{name}", 1, 1)
        self.ddfield = Quantity(self, f"dd{name}", 2, 1)
        self.KG = EoM
    

    


