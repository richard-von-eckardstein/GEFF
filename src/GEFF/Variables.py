from GEFF.bgtypes import BGVal, BGFunc

class BGCollection:
    name = ""
    varnames = []
    dim = 0

    def __init__(self, name, type):
        self.name = name
        try:
            assert type in ["dynamical", "static", "constant"]
        except AssertionError:
            raise TypeError("'type' must be either 'dynamical', 'static' 'function' or 'constant'")
        else:
            self.type = type

        #self.var = self.Define()

    def Define(self):
        return {}

class ScalarField(BGCollection):
    def __init__(self, name):
        self.varnames = [name, f"d{name}"]
        super().__init__(name, "dynamical")

    def Define(self):
        field = BGVal(self.name, 0, 1)
        dfield = BGVal(f"d{self.name}", 1, 1)
        return {field, dfield}
    
class ScalarPotential(BGCollection):
    def __init__(self, name, fields):
        self.varnames = [name, f"d{name}"]
        self.fields = fields
        super().__init__(name, "function")

    def Define(self):
        vars = []
        for field in self.fields:
            assert isinstance(field, ScalarField)
            vars.append( field.Define()[0] )
        
        V = BGFunc(self.varnames[0], vars, 2, 2)
        dV = BGFunc(self.varnames[1], vars, 2, 1)
        return {V, dV}
    
        
class GaugeField(BGCollection):
    def __init__(self, Ename="E", Bname="B", Gname="G", suffix=""):
        self.varnames = [f"{Ename}{suffix}", f"{Bname}{suffix}", f"{Gname}{suffix}"]
        super().__init__(suffix, "dynamical")
        

    def Define(self):
        E = BGVal(self.varnames[0], 4, 0)
        B = BGVal(self.varnames[1], 4, 0)
        G = BGVal(self.varnames[2], 4, 0)
        return {E, B, G}




            

            


    


