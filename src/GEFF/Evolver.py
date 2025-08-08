from GEFF.BGTypes import Quantity, BGSystem



class Evolver:
    def __init__(self, variables, H0, MP):
        quantities=[]
        inds = []
        for var in variables:
            newquantities = var.CreateQuantities()
            if hasattr(var, "EoM"):
                
                setattr(self, var.name, var)
                setattr(self, var)
                

        self.sys = BGSystem.__init__(quantities, H0, MP)

    def Evolve(self, obj):
        return self.EoMs[obj]