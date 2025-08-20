from GEFF.BGTypes import BGSystem
import numpy as np



class NameIndexTranslator:
    def __init__(self, variables, gaugefields, ntr):
        self.variables = variables
        self.gaugefields = gaugefields

        

    
    
    

#Step 1: have array, go through names, retrieve array elements associated with names -> give name, get index
#Step 2: Stuff
#Step 3: have names, assign index to name, create array

#For Step 1 and 2, I know the names. Do I ever need to retrieve names based on indices? No...?


def Evolver(sys, dynamical_variables, gaugefield_variables):
    class Evolver(BGSystem):
        varnames = {"dynamical":dynamical_variables, "GF": gaugefield_variables}
        def __init__(self, initialdata):
            super().from_system(sys)
            pass

class BaseEvolver(BGSystem):
    def __init__(self, dynamical_variables, gaugefields, InitialData, settings):
        super().from_system(InitialData)
        self.Translator = NameIndexTranslator(dynamical_variables, gaugefields)

        var_to_ind = {}
        for index, key in enumerate(dynamical_variables):
            var_to_ind[key] = (index, index+1)

        lastindex = index
        towersize = (settings.ntr+1)
        space =  3*towersize

        for index, key in enumerate(gaugefields):
            start = index*space + lastindex
            end = start + space
            var_to_ind[key] = (start, end)

        self.dydt = np.zeros( (end) )

        self.varnames = {"dynam":dynamical_variables, "GF": gaugefields}

        self.GFshape = (towersize, 3)

        self.GF = dict( zip(gaugefields, [ np.zeros( space ).reshape(self.GFshape) for g in gaugefields ] ) )

    def GetRange( self, name ):
        return self.var_to_ind[name]

    def Evolve(self, key):
        def TimeDerivative(value):
            start, end = self.GetRange( key )
            self.dydt[start:end] = value.flatten()
            return
        
        return TimeDerivative

    def Update(self, t, y):
        self.t.SetValue(t)

        for key in self.varnames["dynam"]:
            start, end = self.GetRange( key )
            getattr(self, key).SetValue( y[start])

        for key in self.varnames["GF"]:
            start, end = self.GetRange( key )
            self.GF[key] = y[start:end].reshape(self.GFshape)
        
        return
    
def ODE(self, t, y):
    self.Evolver.Update(t, y)
    self.StaticVars()
    self.EoM()
    return self.Evolver.dydt


class Solution(BGSystem):
    pass


class Solver:
    pass

