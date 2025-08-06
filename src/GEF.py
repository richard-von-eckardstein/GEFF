import pandas as pd
import numpy as np

from src.BGQuantities import DefaultVariables
from src.BGQuantities.BGTypes import BGSystem, System, Val, Func

from src.Solver.GEFSolver import GEFSolver

import importlib.util as util
import os

from types import NoneType

def LoadModel(name : str, settings : dict):
    """
    Import and execute a module containg a GEF model.

    Parameters
    ----------
    modelname : str
        the name of the GEF model 
    settings : dict
        a dictionary containing the model settings

    Returns
    -------
    ModuleType
        the executed module
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"Models/{name}.py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(name, modelpath)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        for key, item in settings.items():
            try:
                mod.modelSettings[key] = item
            except AttributeError:
                print(f"Ignoring unknown model setting '{key}'.")

        return mod
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found under '{modelpath}'")
    
  
class BaseGEF(System):
    """
    This class is the primary interface for the GEF. It's main function is to create the GEFSolver according to model-specification and to store the results of the GEF.
    Following a successful run, it contains all information about the evolution of the time-dependent background as specified by the model-file.
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
    MbM = None
    Solver = None

    def __init__(
                self, H0, MP, GEFData: NoneType|str = None, ModeData: NoneType|str = None
                ):

        super().__init__(H0, MP)

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

        #TODO what to do about setttings?
        """        #Add any additional settings if applicable
        if isinstance(self.settings, dict):
            for setting in self.settings.items():
                string += f"{setting[0]} : {setting[1]}, """
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
        quantities.update(DefaultVariables.spacetime)
        quantities.update(DefaultVariables.inflaton)
        quantities.update(DefaultVariables.gaugefield)
        quantities.update(DefaultVariables.auxiliary)
        quantities.update(DefaultVariables.inflatonpotential)
        quantities.update(DefaultVariables.coupling)
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

    def LoadGEFData(self, path : NoneType|str=None):
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

    def SaveGEFData(self, path : NoneType|str=None):
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



def GEFModel(modelname, settings):
    #compile the model file
    model = LoadModel(modelname, settings)

    quantities = model.quantities
    quantities = quantities.update(("spacetime",DefaultVariables.spacetime))

    input_signature = {key : [q.name for q in item] for key, item in model.input.items()}

    gaugefields = model.gaugefields

    #These are used 
    dynamical_variables = [q.name for q in quantities["dynamical"]]
    dynamical_dict = {"dynamical":dynamical_variables, "GF": gaugefields.keys()}

    sys = BGSystem(set(quantities.Values()))

    class GEFModel(sys):
        #Add information about the mode-by-mode solver
        MbM = model.MbM
        Solver = GEFSolver( dynamical_dict, model.EoM, model.events, model.MbM )

        def __init__(
                    self, constants : dict, iniVals: dict, Funcs: dict,
                    GEFData: NoneType|str = None, ModeData: NoneType|str = None
                    ):
            
            self.__CheckInput(constants, "constant")
            self.__CheckInput(iniVals, "dynamic")
            self.__CheckInput(Funcs, "functions")

            H0, MP = model.ParseInput(constants, iniVals, Funcs)

            indic = constants
            indic.update(iniVals)
            indic.update(Funcs)
            
            super().FromDic(indic, H0, MP)

            #Add 
            for const, item in constants.items():
                self.Initialise(const)(item)
            
            for func, item in Funcs.items():
                self.Initialise(func)(item)

            self.GEFData = GEFData
            self.ModeData = ModeData

        @classmethod
        def ListInput(cls):
            print("This 'GEF' model requires the following input: \n")
            print(f"Constants: {input_signature["constant"]}")
            print(f"Initial conditions: {input_signature["dynamic"]}")
            print(f"Functions: {input_signature["functions"]}")
            return

        def __CheckInput(self, inputdata, inputtype):
            for val in input_signature[inputtype]:
                try:
                    assert val in inputdata.keys()
                except AssertionError:
                    raise Exception(f"Missing input: '{val}'")     


           

    return GEFModel