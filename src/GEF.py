import pandas as pd
import numpy as np

from src.BGQuantities import DefaultQuantities
from src.BGQuantities.BGTypes import BGSystem, Val, Func

from src.Solver.GEFSolver import GEFSolver

import importlib.util as util
import os

def ModelLoader(modelname : str):
    """
    Import and execute a module containg a GEF model.

    Parameters
    ----------
    modelname : str
        the name of the GEF model 

    Returns
    -------
    ModuleType
        the executed module
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"Models/{modelname}.py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(modelname, modelpath)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found under '{modelpath}'")
        
def SpecifyModelSettings(model, settings : dict={}):
    """
    Update model settings of a GEF model according user specifications 

    Parameters
    ----------
    modelname : ModuleType
        an executed GEF-model module
    settings : dict
        a dictionary containing the new model settings
    """

    for key, item in settings.items():
        try:
            model.modelSettings[key] = item
        except AttributeError:
            print(f"Ignoring unknown model setting '{key}'.")
    
    return

class GEF(BGSystem):
    """
    This class is the primary interface for the GEF. It's main function is to create the GEFSolver according to model-specification and to store the results of the GEF.
    Following a succesful run, it contains all information about the evolution of the time-dependent background as specified by the model-file.
    This information can be passed to various useful tools, for example, computing the gauge-field spectrum, the tensor-power spectrum, and the GW-spectrum.
    The GEF subclasses BGSystem and inherits all its functionalities.
    
    Attributes
    ----------
    MbM : ModeByMode or ModeSolver
        The mode-by-mode class associated to the current GEF-model
    Solver : GEFSolver
        The GEFSolver-instance used to solve the GEF equations

    Methods
    -------
    LoadGEFData()
        Load data and store its results in the current GEF instance.
    SaveGEFData()
        Save the data in the current GEF instance in an ouput file.
    SetUnits()
        Switch the GEF instance between numerical units and physical units
    GetUnits()
        Return a boolean indicating if the GEF is set to physical units

    Example 1 (Initialisation)
    --------------------------
    >>> import numpy as np
    ...
    >>> beta = 20 #Define the axion--gauge field coupling strength
    >>> m = 6e-6 #inflaton mass in Mpl
    ...
    >>> phi = 15.55 #inflaton field value in Mpl
    >>> dphi = -np.sqrt(2/3)*m #inflaton velocity (slow-roll attractor)
    >>> iniVals = {"phi":phi, "dphi":dphi}
    ...
    >>> V = lambda x: 0.5*m**2*x**2 #define the inflaton potential
    >>> dV = lambda x: m**2*x #define the potential derivative
    >>> Funcs  = {"V":V, "dV":dV}
    ...
    >>> G = GEF("Classic", beta=20, iniVals=iniVals, Funcs=Funcs)

    Example 2 (Solving the GEF equations)
    -------------------------------------  
    >>> ntr = 100 #the desired value for truncating gauge-field bilinear tower
    ...
    #Solve the GEF-equations and perform Mode-By-Mode comparison to check convergence
    >>> sol = G.Solver.RunGEF(tend=120, ntr=ntr, atol=1e-20, rtol=1e-6, nmodes=500) 
    >>> G.Solver.ParseArrToUnitsSystem(sol.t, sol.y, G) #Store results in GEF-instance
    ...
    #store the results of the GEF in a file under "Path/To/Some/Output/Directory/File.dat"
    >>> G.SaveGEFData("Path/To/Some/Output/Directory/File.dat")

    Example 3 (Accessing GEF results)
    ---------------------------------
    >>> import matplotlib.pyplot as plt
    ...
    >>> G.LoadGEFData("Path/To/Some/Input/File.dat") #Load data stored under "Path/To/Some/Input/File.dat" 
    ...
    #Retrieve a list of all values stored in the current GEF instance
    >>> print(G.ValueNames())
    ...
    >>> plt.plot(G.N, G.E) #plot the evolution of the electric field expectation value E^2
    >>> plt.show()
    """
 
    def __init__(
                self, model: str, beta: float, iniVals: dict, Funcs: dict,
                userSettings: dict = {}, GEFData: None|str = None, ModeData: None|str = None
                ):
        
        #Get Model attributes
        model = ModelLoader(model)

        #Configure model settings
        SpecifyModelSettings(model, userSettings)

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

        #Get model-specific mode-by-mode class
        self.MbM = model.ModeByMode

        #Create the GEF solver
        self.__SetupGEFSolver(model, iniVals, Funcs)

        #Add information about file paths
        self.GEFData = GEFData
        self.ModeData = ModeData

        return
    
    def __str__(self):
        """
        Return a string representing the current GEF instance.

        Returns
        -------
        str
            The GEF instance represented as a string.
        """

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
    def __DefineQuantities(modelSpecific):
        """
        Create a dictionary of BGVals and BGFuncs used to initialise the GEF.

        Parameters
        ----------
        modelSpecific : dict of BGVal's and BGFunc's
            specifies the model-specific quantities used by the GEF

        Returns
        -------
        quantities : dict of BGVal's and BGFunc's
            all quantities known by the GEF system.
        """

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
    
    def __SetupGEFSolver(self, model, iniVals : dict, Funcs : dict):
        """
        Configure the GEF-Solver according to a GEF model file

        Parameters
        ----------
        model : ModuleType
            an executed GEF-model module
        iniVals : dict
            a dictionary of values used as initial conditions for the GEF-solver
        Funcs : dict
            a dictionary of functions used to specify model-specific functions like the inflaton potential
        """

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

    def LoadGEFData(self, path=None):
        """
        Load data and store its results in the current GEF instance.

        Parameters
        ----------
        path : None or str
            if None, loads data from self.GEFData, otherwise loads data from the specified path.

        Raises
        ------
        Exception
            if 'path' is None but self.GEFData is also None
        FileNotFoundError
            if no file is found at 'path'
        AttributeError
            if the file contains a column labeled by a key which does not match any GEF-value name.
        """

        if path==None:
            path=self.GEFData
        else:
            self.GEFData=path

        #Check if GEF has a file path associated with it
        if path == None:
            raise Exception("You did not specify the file from which to load the GEF data. Set 'GEFData' to the file's path from which you want to load your data.")
        else:
            #Check if file exists
            try:
                #Load from file
                input_df = pd.read_table(path, sep=",")
            except FileNotFoundError:
                raise FileNotFoundError(f"No file found under '{path}'")
        
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

    def SaveGEFData(self, path=None):
        """
        Save the data in the current GEF instance in an ouput file.

        Parameters
        ----------
        path : str
            if None, stores data in self.GEFData, otherwise stores data in the specified file.

        Raises
        ------
        Exception
            if 'path' is None but self.GEFData is also None
        
        """
        if path==None:
            path=self.GEFData
        else:
            self.GEFData=path
        
        if path==None:
            raise Exception("You did not specify the file under which to store the GEF data. Set 'GEFData' to the location where you want to save your data.")

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
