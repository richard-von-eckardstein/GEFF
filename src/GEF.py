import pandas as pd
import numpy as np

from src.BGQuantities import DefaultQuantities
from src.BGQuantities.BGTypes import BGSystem, Val, Func

from src.Solver.GEFSolver import GEFSolver

import importlib.util as util
import os
import warnings
from copy import deepcopy



class GEF(BGSystem):
    """
    This class is the primary interface for the GEF. It's primary function is to create the GEFSolver according to model-specification and to store its results.
    Furthermore, it contains all information about the evolution of the time-dependent background as specified by the model-file. This information can be passed to
    various useful tools, for example, computing the gauge-field spectrum, the tensor-power spectrum, and the GW-spectrum.
    
    Attributes
    ----------
    name : str
        the name of the GEF-model
    beta : float
        Coupling strength of the inflaton to the gauge fields, dI/dphi = beta/Mpl
    GEFData : str
        Path to file where GEF results are stored
    ModeData : str
        Path to file where Mode-by-Mode results are stored

    Methods
    -------
    LoadGEFData
        Load data stored in self.GEFData and use it to initialise the GEF background quantities
    SaveGEFData
        Store the data in the current GEF instance in self.GEFData.

    """
 
    def __init__(
                self, model: str, beta: float, iniVals: dict, Funcs: dict,
                userSettings: dict = {}, GEFData: None|str = None, ModeData: None|str = None
                ):
        
        #Get Model attributes
        model = self.ModelLoader(model)

        #Configure model settings
        self.SpecifyModelSettings(model, userSettings)

        #Set GEF-name
        self.name = model.name

        #Set coupling constant
        self.beta = beta
        
        #Define background quantities
        quantities = self.__DefineQuantities(model.modelQuantities)

        #Compute H0 from initial conditions
        rhoInf = 0.5*iniVals["dphi"]**2 + Funcs["V"](iniVals["phi"])
        rhoEM = 0.
        rhoExtra = [iniVals[rho] for rho in model.modelRhos]

        H0 = np.sqrt( (rhoInf + rhoEM + sum(rhoExtra))/3 )
        MP = 1.

        #Define the GEFClass as a BGSystem using the background quantities, functions and unit conversions
        super().__init__(quantities, H0, MP)

        self.MbM = model.ModeByMode

        #Create the GEF solver class
        self.__SetupGEFSolver(model, iniVals, Funcs)

        #Add information about storage
        self.GEFData = GEFData
        self.ModeData = ModeData

        return
    
    def __str__(self):
        #Add the model information
        string = f"Model: {self.name}, "
        #Add any additional settings if applicable
        if isinstance(self.settings, dict):
            for setting in self.settings.items():
                string += f"{setting[0]} : {setting[1]}, "
        #Add coupling strength
        string += f"beta={self.beta}"
        return string
    
    @staticmethod
    def SpecifyModelSettings(model, settings={}):
        for key, item in settings.items():
            try:
                model.modelSettings[key] = item
            except AttributeError:
                print(f"Ignoring unknown model setting '{key}'.")
        
        return

    @staticmethod
    def ModelLoader(modelname):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modelpath = os.path.join(current_dir, f"Models/{modelname}.py")
        #Check if Model exists
        try:
            #Load ModelAttributes from GEFFile
            spec = util.spec_from_file_location(modelname, modelpath)
            mod  = util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except:
            raise FileNotFoundError(f"No model found under '{modelpath}'")
    
    def __SetupGEFSolver(self, model, iniVals, Funcs):
        for obj in self.ObjectSet():
            if issubclass(obj, Val):
                self.Initialise(obj.name)(0.)
            if issubclass(obj, Func):
                self.Initialise(obj.name)(lambda x: 0.)

        for key, item in iniVals.items():
            self.Initialise(key)(item)
        for key, item in Funcs.items():
            self.Initialise(key)(item)

        if not("dI" in Funcs.keys()):
            self.Initialise("dI")(lambda x: float(self.beta))
        if not("ddI" in Funcs.keys()):
            self.Initialise("ddI")(lambda x: 0.)

        self.SetUnits(False)

        self.Solver = GEFSolver(
                                model.UpdateVals, model.TimeStep, model.Initialise,
                                    model.events, model.ModeByMode, self)
        self.completed=False
        
        return
    
    @staticmethod
    def __DefineQuantities(modelSpecific):
        quantities = set()
        #Get default quantities which are always present in every GEF run
        quantities.update(DefaultQuantities.spacetime)
        quantities.update(DefaultQuantities.inflaton)
        quantities.update(DefaultQuantities.gaugefield)
        quantities.update(DefaultQuantities.auxiliary)
        quantities.update(DefaultQuantities.inflatonpotential)
        quantities.update(DefaultQuantities.coupling)
        quantities.update(modelSpecific)

        return quantities

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

        data = dict(zip(input_df.columns[1:],input_df.values[:,1:].T))

        #Befor starting to load, check that the file is compatible with the GEF setup.
        names = self.ObjectNames()
        for key in data.keys():
            if not(key in names):
                raise AttributeError(f"The data table you tried to load contains an unkown quantity: '{key}'")
        
        #Store current units to switch back to later
        units=self.GetUnits()

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.SetUnits(False)
        #Load data into background-value attributes
        for key, values in data.items():
            self.Initialise(key)(values)
        self.SetUnits(units)
        self.completed=True

        return

    def SaveGEFData(self):
        if self.GEFData==None:
            print("You did not specify the file under which to store the GEF data. Set 'GEFData' to the location where you want to save your data.")
        else:
            valuelist = self.ValueList()
            if valuelist==[]:
                print("No data to store.")
                return
            else:
                path = self.GEFData

                #Create a dictionary used to create pandas data table
                dic = {}

                #remember the original units of the GEF
                units=self.GetUnits()

                #Data is always stored unitless
                self.SetUnits(False)

                for val in valuelist:
                    key = val.name
                    dic[key] = val.value
                
                #Create pandas data frame and store the dictionary under the user-specified path
                output_df = pd.DataFrame(dic)  
                output_df.to_csv(path)

                #after storing data, restore original units
                self.SetUnits(units)
        return
    
    def EndOfInflation(x, tol=1e-4):
        pass