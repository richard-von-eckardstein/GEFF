import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from src.BGQuantities import DefaultQuantities
from src.BGQuantities.BGTypes import BGVal, BGFunc, BGSystem
import importlib.util as util
import os
import warnings

def ModelLoader(modelname):
    modelpath = os.path.join("./src/Models/", modelname+".py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(modelname, modelpath)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except:
        raise FileNotFoundError(f"No model found under '{modelpath}'")

class GEF(BGSystem):
    """
    A class used to solve the GEF equations given a set of initial conditions. The class also comes with several useful utility functions.
    
    ...
    
    Attributes
    ----------
    
    beta : float
        Coupling strength of the inflaton to the gauge fields, I_2(phi) = beta/Mpl
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
                userSettings: dict = {}, GEFData: None|str = None, ModeData: None|str = None
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
        model = ModelLoader(model)

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

        necessarykeys = []

        #initialise GEFValues
        for key, item in quantities.items():    
            #Check if a complete GEF value requires knowledge of this value.
            item.setdefault("optional", True) #If the "optional flag is not set, it is assumed the key is optional"
            if not(item["optional"]):
                necessarykeys.append(key)
            #default key is not longer needed from here on out
            item.pop("default")
    
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
                    if key in necessarykeys:
                        warnings.warn(f"No default value set for '{key}'")
                    
            self.__necessarykeys = necessarykeys
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
    
    def PrintNecessaryKeys(self):
        print(f"Necessary keys for this GEF-setup are:\n{self.__necessarykeys}")

    #ToDo
    def LoadGEFData(self):
        #Check if GEF has a file path associated with it
        if self.GEFData == None:
            print("You did not specify the file from which to load the GEF data. Set 'GEFData' to the file's path from which you want to load your data.")
            return
        else:
            #Check if file exists
            file = self.GEFData
            try:
                #Load from file
                input_df = pd.read_table(file, sep=",")
            except FileNotFoundError:
                raise FileNotFoundError(f"No file found under '{file}'")
        
        #Dictionary for easy access using keys
        data = dict(zip(input_df.columns[1:],input_df.values[1:,1:].T))

        #Check if data file is complete
        for key in self.__necessarykeys:
            if key not in data.keys():
                raise KeyError(f"The file you provided does not contain information on the parameter'{key}'. Please provide a complete data file")

        #Befor starting to load, check that the file is compatible with the GEF setup.
        for key in data.keys():
            try:
                obj = getattr(self, key)
                assert isinstance(obj, BGVal)
            except AttributeError:
                raise AttributeError(f"The data table you tried to load contains an unkown value: '{key}'")
            except AssertionError:
                raise AttributeError(f"'{key}' is not a known value")
        
        #Store current units to switch back to later
        units=self.units

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.SetUnits(False)
        #Load data into background-value attributes
        for key, values in data.items():
            obj = getattr(self, key)
            obj.SetVal(values)
        self.SetUnits(units)

        return

    def SaveData(self):
        if self.GEFData==None:
            print("You did not specify the file under which to store the GEF data. Set 'GEFData' to the location where you want to save your data.")
        else:
            path = self.GEFData

            #Create a dictionary used to create pandas data table
            dic = {}

            #remember the original units of the GEF
            units = self.units

            for key in vars(self):
                obj = getattr(self, key)
                #Only store BGVal instances
                if isinstance(obj, BGVal):
                    #Make sure to not store unitialised BGVal instances
                    if isinstance(obj.value, type(None)):
                        #If a necessary key is not initialised, the data cannot be stored. Unitialised optional keys are ignored.
                        if (key in self.__necessarykeys):
                            raise ValueError(f"Incomplete data. No values assigned to '{key}'.")
                    else:
                        #Add the quantities value to the dictionary
                        obj.SetUnits(False)
                        dic[key] = obj.value
                        obj.SetUnits(units)

            #Create pandas data frame and store the dictionary under the user-specified path
            output_df = pd.DataFrame(dic)  
            output_df.to_csv(path)
        return
    
    def EndOfInflation(x, tol=1e-4):
        pass