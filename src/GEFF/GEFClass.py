import pandas as pd
import numpy as np

from GEFF import DefaultQuantities
from GEFF.BGTypes import BGSystem, Val, Func

from GEFF.GEFSolver import GEFSolver

import importlib.util as util
import os

from numbers import Number
from types import NoneType

class MissingInputError(Exception):
    pass

class NegativeEnergyError(Exception):
    pass

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
    

def CompileModel(modelname, settings):
    #compile the model file
    model = LoadModel(modelname, settings)

    #import quantities dictionary
    Q_dic = model.quantities

    #import information on input and how to handle it
    input_signature = model.input
    input_handler = model.ParseInput

    #import information for solver:
    init_func = model.Initialise
    staticVariable_func = model.UpdateVals#model.ComputeStaticvariables
    EoM_func = model.TimeStep
    event_list = model.events

    #import Mode-By-Mode class:
    MbM_solver = model.MbM

    return Q_dic, (input_signature, input_handler), (staticVariable_func, EoM_func, init_func, event_list, MbM_solver), MbM_solver
    

def ModelSetup(modelname, settings):
    def GEF_decorator(cls):
        quantity_info, input_info, solver_info, MbM_info = CompileModel(modelname, settings)
        
        cls.ObjectClassification = { key:{i.name for i in item} for key, item in quantity_info.items()}
        quantity_info.pop("gaugefields")
        cls.KnownObjects = set().union(*quantity_info.values())
        

        cls.InputSignature = input_info[0]
        cls.InputHandler = staticmethod(input_info[1])
        cls.GEFSolver = GEFSolver(*solver_info)
        cls.ModeSolver = MbM_info
        return cls
    return GEF_decorator

class GEFType:
    def __new__(cls, *args, **kwargs):
        #Here, I can ensure that the input from the model really compiles, i.e.,
        #that all class attributes are as intended
        return super().__new__(cls, *args, **kwargs)

@ModelSetup("Classic", {})
class BaseGEF(BGSystem):
    """
    This class is the primary interface for the GEF. It's main function is to create the GEFSolver according to model-specification and to store the results of the GEF.
    Following a successful run, it contains all information about the evolution of the time-dependent background as specified by the model-file.
    This information can be passed to various useful tools, for example, computing the gauge-field spectrum, the tensor-power spectrum, and the GW-spectrum.
    The GEF subclasses BGSystem and inherits all its functionalities.
    
    Attributes
    ----------
    ModeSolver : ModeByMode subclass
        The mode-by-mode class associated to the current GEF-model
    GEFSolver : GEFSolver
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
    >>> sol = G.GEFSolver.RunGEF(tend=120, ntr=ntr, atol=1e-20, rtol=1e-6, nmodes=500) 
    >>> G.GEFSolver.ParseArrToUnitsSystem(sol.t, sol.y, G) #Store results in GEF-instance
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
                self, consts : dict, iniVals : dict, Funcs : dict, 
                GEFData: NoneType|str = None, ModeData: NoneType|str = None
                ):
        
        user_input = {"constants":consts, "initial data":iniVals, "functions":Funcs}
        #Check that all necessary input is present and that its data type is correct
        for inputtype, inputdic  in user_input.items():
            self.__CheckInput(inputdic, inputtype)

        H0, MP = self.InputHandler(*user_input.values())

        super().__init__(self.KnownObjects, H0, MP)

        #Add initial data to BGSystem
        for name, constant in user_input["constants"].items():
            self.Initialise(name)(constant)
        for name, function in user_input["functions"].items():
            self.Initialise(name)(function)
        for name, value in user_input["initial data"].items():
            self.Initialise(name)(value)

        #Initialise the other values with dummy variables.
        for name in self.ObjectNames():
            if name not in (self.ValueNames() + self.FunctionNames()):
                self.Initialise(name)(0)

        self.GEFSolver.SetIniVals(self)

        #Add information about file paths
        self.GEFData = GEFData
        self.ModeData = ModeData

    @classmethod
    def PrintInput(cls):
        print("This 'GEF' model requires the following input:")
        for key, item in cls.InputSignature.items():
            print(f"\t {key.capitalize()}: {item}")
        print("\n")
        return

    def __CheckInput(self, inputdata : dict, inputtype : str):
        for key in self.InputSignature[inputtype]:
            try:
                assert key in inputdata.keys()
            except AssertionError:
                raise MissingInputError(f"Missing input in '{inputtype}': '{key}'")

            if inputtype == "functions":
                try:
                    assert callable(inputdata[key])
                except AssertionError:
                    raise TypeError(f"Input functions must be callable.")
            else:
                try:
                    assert isinstance(inputdata[key], Number)
                except AssertionError:
                    raise TypeError(f"Input {inputtype} is '{type(inputdata[key])}' but should be 'Number' type.")
        return
    
    def RunGEF(self, ntr, tend, nmodes=500, printstats=True, **Kwargs):

        #Configuring GEFSolver
        self.GEFSolver.ntr=ntr
        self.GEFSolver.tend=tend
        GEFKwargs = {setting : Kwargs[setting] for setting in self.GEFSolver.settings if setting in Kwargs}        
        self.GEFSolver.UpdateSettings(**GEFKwargs)

        #Configuring ModeSolver
        MbMattempts = Kwargs.get("MbMattempts", 5)
        thinning = Kwargs.get("thinning", 5)
        errthr = Kwargs.get("errthr", 0.025)
        resumeMode = Kwargs.get("resumeMode", True)
        method = Kwargs.get("method", "simpson")
        selfcorrmethod = Kwargs.get("selfcorrmethod", "simpson")

        MbMKwargs = {"epsabs":self.GEFSolver.settings["atol"], "epsrel":self.GEFSolver.settings["rtol"]}

        rtol = self.GEFSolver.settings["rtol"]

        done=False
        sol = None
        attempt=0

        while not(done) and attempt<MbMattempts:
            attempt +=1
            #This can be taken care of internally
            solnew, vals = self.GEFSolver.GEFAlgorithm()
            sol = self.GEFSolver.UpdateSol(sol, solnew)
            self.GEFSolver.ParseArrToUnitSystem(sol.t, sol.y, vals)

            if nmodes!=None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeSolver(vals)
                rtol = self.GEFSolver.settings["rtol"]

                if resumeMode and attempt > 1:
                    spec = MbM.UpdateSpectrum(spec, treinit, rtol=rtol)
                else:
                    spec = MbM.ComputeModeSpectrum(nmodes, rtol=rtol)
                print("Performing mode-by-mode comparison with GEF results.")

                agreement, ReInitSpec = self.GEFSolver.ModeByModeCrossCheck(spec, vals, errthr=errthr, thinning=thinning, method=selfcorrmethod, **MbMKwargs)

                if agreement:
                    if len(sol.events["Negative energies"]["t"]) > 0: 
                        raise NegativeEnergyError(
                            "The GEF solution claims convergence on a negative energy solution." \
                        "Discarding the solution. Try lowering the mode-by-mode error tolerance."
                        )
                    else:
                        print(f"The mode-by-mode comparison indicates a convergent GEF run.")
                        done=True
                
                else:
                    Nreinit = np.round(ReInitSpec["N"], 1)
                    treinit = np.round(ReInitSpec["t"], 1)
                    print(f"Attempting to solve GEF using self-correction starting from t={treinit}, N={Nreinit}.")

                    self.InitialConditions = self.GEFSolver.InitialiseFromMbM(sol, ReInitSpec, method, **MbMKwargs)
                
            else:
                spec=None
                done=True
        
        if done:
            print("GEF run successfully completed.")
            if printstats: PrintSummary(sol)
            return sol, spec
        else:
            raise RuntimeError(f"GEF did not complete after {attempt} attempts.")

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
            storeables = set().union(
                                     self.ObjectClassification["time"],
                                     self.ObjectClassification["dynamical"],
                                     self.ObjectClassification["static"]
                                    )           
            #Check that all dynamic and derived quantities are initialised in this GEF instance
            if not( storeables.issubset(set( self.ValueNames() )) ):
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

                for key in storeables:
                    dic[key] = getattr(self, key).value
                
                #Create pandas data frame and store the dictionary under the user-specified path
                output_df = pd.DataFrame(dic)  
                output_df.to_csv(path)

                #after storing data, restore original units
                self.SetUnits(units)
        return



def GEF(modelname, settings):

    @ModelSetup(modelname, settings)
    class GEF(BaseGEF):
        pass

    return GEF


def PrintSummary(sol):
    print("The run terminated with the following statistics:")
    for attr in sol.keys():
        if attr not in ["y", "t", "y_events", "t_events", "sol", "events"]:
            print(rf"{attr} : {getattr(sol, attr)}")
    events = sol.events
    if len(events.keys())==0:
        print("No events occured during the run")
    else:
        print("The following events occured during the run:")
        for event in events.keys():
            time = events[event]["t"]
            efold = events[event]["N"]
            print(rf"{event} at t={time} or N={efold}")
    return