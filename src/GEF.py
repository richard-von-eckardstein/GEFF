import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from src.BGQuantities import DefaultQuantities
from src.BGQuantities.BGTypes import BGVal, BGFunc, BGSystem
import importlib.util as util
import os

def ModelBuilder(modelname):
    modelpath = os.path.join("./src/Models/", modelname+".py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(modelname, modelpath)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except:
        raise FileNotFoundError(f"No model found under {modelpath}")
    
class MissingInputError(Exception):
    pass


class GEF(BGSystem):
    """
    A class used to solve the GEF equations given a set of initial conditions. The class also comes with several useful utility functions.
    
    ...
    
    Attributes
    ----------
    
    beta : float
        Coupling strength of the inflaton to the gauge fields, I_2(phi) = beta/Mpl
    approx : Boolean
        Are Whittaker functions for xi>4 evaluated explicitly or using approximate expressions with rel error < 1e-4
    units : Boolean
        Are all quantities treated as dimensionful or dimensioneless?
    H0 : float
        initial value of the Hubble rate in the dimensional unit system. Used for unit conversion.
    Mpl : float
        value of the Planck mass in the dimensional unit system. Used for unit conversion.
    GEFData : None | str
        Path to file where GEF results are stored
    ModeData : None | str
        Path to file where Mode By Mode results are stored
        
    ...
    
    Methods
    -------
    
    ...
    """
 
    def __init__(
                self, model: str, beta: float, iniVals: dict, Funcs: dict,
                userSettings: dict = {}, GEFData: None|str = None, ModeData: None|str = None, approx: bool = True
                ):
        #...

        #At initialisation, all input is assumed to be in Planck units
        self.units = True

        #Set coupling constant
        self.beta = beta

        #Compute H0 from initial conditions
        H0 = np.sqrt( ( 0.5*iniVals["dphi"]**2 + Funcs["V"](iniVals["phi"]) )/3 )
        MP = 1.

        #Get Model attributes
        model = ModelBuilder(model)

        #Set GEF-name
        self.__name = model.name
        
        #Define background quantities
        valuedic = self.__InitialiseQuantities(model.modelQuantities, iniVals)
        #Define background functions
        functiondic = self.__InitialiseFunctions(model.modelFunctions, Funcs)

        #Define the GEFClass as a BGSystem using the background quantities, functions and unit conversions
        super().__init__(valuedic, functiondic, H0, MP)

        #Configure model settings
        self.__ConfigureModelSettings(model.modelSettings, userSettings)

        #Add information about storage
        self.GEFData = GEFData
        self.ModeData = ModeData
        
        self.approx = approx

        return
    
    def __str__(self):
        string = f"Model: {self.__name}, "
        if isinstance(self.settings, dict):
            for setting in self.settings.items():
                string += f"{setting[0]} : {setting[1]}, "
        string += f"beta={self.beta}"
        return string

    def __ConfigureModelSettings(self, modelSettings, userSettings):
        settings = {}
        for setting in modelSettings.keys():
            try:
                settings[setting] = userSettings[setting]
            except:
                settings[setting] = modelSettings[setting]

        if settings == {}:
            self.settings = None
        else:
            self.settings = settings
        return
                
    
    def __InitialiseQuantities(self, modelSpecific, iniVals):
        #Get default quantities which are always present in every GEF run
        spacetime = DefaultQuantities.spacetime
        inflaton = DefaultQuantities.inflaton
        gaugefield = DefaultQuantities.gaugefield
        auxiliary = DefaultQuantities.auxiliary
        
        #concatenate dictionary and update default values according to model
        quantities = spacetime | inflaton | gaugefield | auxiliary
        quantities.update(modelSpecific)

        #initialise GEFValues
        for key, item in quantities.items():    
            #Add initial data
            if key in iniVals.keys():
                #Initial data from input
                item["value"]=iniVals[key]
            else:
                try:
                    #initial data from default
                    item["value"] = item["default"]
                    item.pop("default")
                except:
                    item["value"] = None
                    print(f"No default value set for {key}")
            #Remove default key from value dictionary
            if hasattr(item, "default"):
                item.pop("default")
        return quantities
    
    def __InitialiseFunctions(self, modelSpecific, Funcs):
        #Get default functions which are always present in every GEF run
        inflatonpotential = DefaultQuantities.inflatonpotential
        coupling = DefaultQuantities.coupling
        coupling["dI"]["func"] = lambda x: self.beta/self.MP
        coupling["ddI"]["func"] = lambda x: 0.

        #concatenate dictionary of functions and update default functions according to model
        functions = inflatonpotential | coupling
        functions.update(modelSpecific)

        #initialise GEFFunctions
        for key, item in functions.items():
            if key in ["dI", "ddI"]:
                #inflaton--gauge field coupling initialised separetly
                func = coupling[key]["func"]
            else:
                #Check if Function is passed by the User via Funcs
                try:
                    func = Funcs[key]
                except:
                    raise KeyError(f"'Funcs' needs to declare the function '{key}'")
            item["func"] = func
            #Define the function as a BGFunc

        return functions
    

    #ToDo   
    def SaveData(x):
        pass

    def LoadData(x):
        pass
    
    def EndOfInflation(x, tol=1e-4):
        pass