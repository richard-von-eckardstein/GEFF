from src.BGQuantities.BGTypes import DefineQuantity
from src.Solver.Evolver import Evolver

        
class ScalarField:
    def __init__(self, name, KG):
        self.name = name
        names = [name, f"d{name}"]
        self.__KG = KG

    def CreateQuantities(self):
        field = DefineQuantity(self.name, 0, 1)
        dfield = DefineQuantity(f"d{self.name}", 1, 1)
        return {field, dfield}
        
class GaugeField:
    def __init__(self, name, EoM):
        pass




            

            


    


