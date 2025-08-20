import pandas as pd
import numpy as np

from GEFF.BGTypes import BGSystem
from GEFF.GEFSolver import GEFSolver

import importlib.util as util
import os

from numbers import Number
from types import NoneType

class MissingInputError(Exception):
    pass

class NegativeEnergyError(Exception):
    pass

def _load_model(name : str, user_settings : dict):
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

        for key, item in user_settings.items():
            try:
                mod.settings[key] = item
            except AttributeError:
                print(f"Ignoring unknown model setting '{key}'.")

        return mod
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found under '{modelpath}'")
    

def _compile_model(modelname, user_settings):
    #compile the model file
    model = _load_model(modelname, user_settings)

    #import quantities dictionary
    q_dict = model.quantities

    #import information on input and how to handle it
    input_signature = model.input
    input_handler = model.parse_input

    #import information for solver:
    init_func = model.initial_conditions
    static_variable_func = model.update_values#model.ComputeStaticvariables
    EoM_func = model.compute_timestep
    event_list = model.events

    #import Mode-By-Mode class:
    MbM_solver = model.MbM

    return q_dict, (input_signature, input_handler), (static_variable_func, EoM_func, init_func, event_list), MbM_solver
    

def _model_setup(model_name, user_settings):
    def GEF_decorator(cls):
        quantity_info, input_info, solver_info, MbM_info = _compile_model(model_name, user_settings)
        
        object_classifier = { key:{i.name for i in item} for key, item in quantity_info.items()}
        cls._object_classification = object_classifier
        
        quantity_info.pop("gaugefields")
        cls._known_objects = set().union(*quantity_info.values())
        

        cls._input_signature = input_info[0]
        cls._input_handler = staticmethod(input_info[1])
        cls.GEFSolver = GEFSolver(*solver_info, variable_dict=object_classifier)
        cls.ModeSolver = MbM_info
        return cls
    return GEF_decorator

class GEFType:
    def __new__(cls, *args, **kwargs):
        #Here, I can ensure that the input from the model really compiles, i.e.,
        #that all class attributes are as intended
        return super().__new__(cls, *args, **kwargs)

@_model_setup("Classic", {})
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
    load_GEFdata()
        Load data and store its results in the current GEF instance.
    save_GEFdata()
        Save the data in the current GEF instance in an ouput file.
    set_units()
        Switch the GEF instance between numerical units and physical units
    get_units()
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
    >>> init_dict = {"phi":phi, "dphi":dphi}
    ...
    >>> V = lambda x: 0.5*m**2*x**2 #define the inflaton potential
    >>> dV = lambda x: m**2*x #define the potential derivative
    >>> init_funcs  = {"V":V, "dV":dV}
    ...
    >>> G = GEF("Classic", beta=20, init_dict=init_dict, init_funcs=init_funcs)

    Example 2 (Solving the GEF equations)
    -------------------------------------  
    >>> ntr = 100 #the desired value for truncating gauge-field bilinear tower
    ...
    #Solve the GEF-equations and perform Mode-By-Mode comparison to check convergence
    >>> sol = G.GEFSolver.RunGEF(tend=120, ntr=ntr, atol=1e-20, rtol=1e-6, nmodes=500) 
    >>> G.GEFSolver.ParseArrToUnitsSystem(sol.t, sol.y, G) #Store results in GEF-instance
    ...
    #store the results of the GEF in a file under "Path/To/Some/Output/Directory/File.dat"
    >>> G.save_GEFdata("Path/To/Some/Output/Directory/File.dat")

    Example 3 (Accessing GEF results)
    ---------------------------------
    >>> import matplotlib.pyplot as plt
    ...
    >>> G.load_GEFdata("Path/To/Some/Input/File.dat") #Load data stored under "Path/To/Some/Input/File.dat" 
    ...
    #Retrieve a list of all values stored in the current GEF instance
    >>> print(G.value_names())
    ...
    >>> plt.plot(G.N, G.E) #plot the evolution of the electric field expectation value E^2
    >>> plt.show()
    """

    def __init__(
                self, consts : dict, init_dict : dict, init_funcs : dict, 
                GEFdata: NoneType|str = None, MbMdata: NoneType|str = None
                ):
        
        user_input = {"constants":consts, "initial data":init_dict, "functions":init_funcs}
        #Check that all necessary input is present and that its data type is correct
        for input_type, input_dict  in user_input.items():
            self._check_input(input_dict, input_type)

        H0, MP = self._input_handler(*user_input.values())

        super().__init__(self._known_objects, H0, MP)

        #Add initial data to BGSystem
        for name, constant in user_input["constants"].items():
            self.initialise(name)(constant)
        for name, function in user_input["functions"].items():
            self.initialise(name)(function)
        for name, value in user_input["initial data"].items():
            self.initialise(name)(value)

        #initialise the other values with dummy variables.
        for name in self.object_names():
            if name not in (self.value_names() + self.function_names()):
                self.initialise(name)(0)

        self.GEFSolver.set_init_vals(self)

        #Add information about file paths
        self.GEFdata = GEFdata
        self.MbMdata = MbMdata

    @classmethod
    def print_input(cls):
        print("This 'GEF' model requires the following input:")
        for key, item in cls._input_signature.items():
            print(f"\t {key.capitalize()}: {item}")
        return

    def _check_input(self, input_data : dict, input_type : str):
        for key in self._input_signature[input_type]:
            try:
                assert key in input_data.keys()
            except AssertionError:
                raise MissingInputError(f"Missing input in '{input_type}': '{key}'")

            if input_type == "functions":
                try:
                    assert callable(input_data[key])
                except AssertionError:
                    raise TypeError("Input 'functions' must be callable.")
            else:
                try:
                    assert isinstance(input_data[key], Number)
                except AssertionError:
                    raise TypeError(f"Input '{input_type}' is '{type(input_data[key])}' but should be 'Number' type.")
        return
    
    def run(self, ntr, tend, nmodes=500, print_stats=True, **kwargs):

        #Configuring GEFSolver
        self.GEFSolver.ntr=ntr
        self.GEFSolver.tend=tend
        GEFkwargs = {setting : kwargs[setting] for setting in self.GEFSolver.settings if setting in kwargs}        
        self.GEFSolver.update_settings(**GEFkwargs)

        #Configuring ModeSolver
        MbMattempts = kwargs.get("MbMattempts", 5)
        binning = kwargs.get("binning", 5)
        errthr = kwargs.get("errthr", 0.025)
        resumeMbM = kwargs.get("resumeMbM", True)
        method = kwargs.get("method", "simpson")
        selfcorrmethod = kwargs.get("selfcorrmethod", "simpson")

        MbMkwargs = {"epsabs":self.GEFSolver.settings["atol"], "epsrel":self.GEFSolver.settings["rtol"]}

        rtol = self.GEFSolver.settings["rtol"]

        done=False
        sol = None
        spec = None
        t_reinit = 0.
        attempt=0

        while not(done) and attempt<MbMattempts:
            attempt +=1
            #This can be taken care of internally
            solnew, vals = self.GEFSolver.compute_GEF_solution()
            sol = self.GEFSolver.update_sol(sol, solnew)
            self.GEFSolver.parse_arr_to_sys(sol.t, sol.y, vals)

            if nmodes is not None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeSolver(vals)
                rtol = self.GEFSolver.settings["rtol"]

                if resumeMbM and attempt > 1:
                    #How to fix?
                    spec = MbM.update_spectrum(spec, t_reinit, rtol=rtol)
                else:
                    spec = MbM.compute_spectrum(nmodes, rtol=rtol)
                print("Performing mode-by-mode comparison with GEF results.")

                agreement, reinit_spec = self.GEFSolver.MbMcrosscheck(spec, vals, errthr=errthr, binning=binning, method=selfcorrmethod, **MbMkwargs)

                if agreement:
                    print("The mode-by-mode comparison indicates a convergent GEF run.\n")
                    done=True
                
                else:
                    t_reinit = reinit_spec["t"]
                    print(f"Attempting to solve GEF using self-correction starting from t={np.round(reinit_spec['t'], 1)}, N={np.round(reinit_spec['N'], 1)}.\n")

                    self.GEFSolver.compute_initial_conditions = self.GEFSolver.initialise_from_MbM(sol, reinit_spec, method, **MbMkwargs)
                
            else:
                done=True
        
        if done:
            
            if print_stats:
                print_summary(sol)
            if sol.success:
                print("\nStoring results in GEF instance.")
                self.GEFSolver.parse_arr_to_sys(sol.t, sol.y, self)
            else:
                print("The run terminated on with an error, check output for details.")
            return sol, spec
        
        else:
            raise RuntimeError(f"GEF did not complete after {attempt} attempts.")

    def load_GEFdata(self, path : NoneType|str=None):
        """
        Load data and store its results in the current GEF instance.

        Parameters
        ----------
        path : None or str
            if None, loads data from self.GEFdata, otherwise loads data from the specified path.

        Raises
        ------
        Exception
            if 'path' is None but self.GEFdata is also None
        FileNotFoundError
            if no file is found at 'path'
        AttributeError
            if the file contains a column labeled by a key which does not match any GEF-value name.
        """

        if path is None:
            path=self.GEFdata
        else:
            self.GEFdata=path

        #Check if GEF has a file path associated with it
        if path is None:
            raise Exception("You did not specify the file from which to load the GEF data. Set 'GEFdata' to the file's path from which you want to load your data.")
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
        names = self.object_names()
        for key in data.keys():
            if key not in names:
                raise AttributeError(f"The data table you tried to load contains an unkown quantity: '{key}'")
        
        #Store current units to switch back to later
        units=self.get_units()

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.set_units(False)
        #Load data into background-value attributes
        for key, values in data.items():
            self.initialise(key)(values)
        self.set_units(units)
        self.completed=True

        return

    def save_GEFdata(self, path : NoneType|str=None):
        """
        Save the data in the current GEF instance in an ouput file.

        Parameters
        ----------
        path : str
            if None, stores data in self.GEFdata, otherwise stores data in the specified file.

        Raises
        ------
        Exception
            if 'path' is None but self.GEFdata is also None
        
        """
        if path is None:
            path=self.GEFdata
        else:
            self.GEFdata=path
        
        if path is None:
            raise Exception("You did not specify the file under which to store the GEF data. Set 'GEFdata' to the location where you want to save your data.")

        else:
            storeables = set().union(
                                     self._object_classification["time"],
                                     self._object_classification["dynamical"],
                                     self._object_classification["static"]
                                    )           
            #Check that all dynamic and derived quantities are initialised in this GEF instance
            if not( storeables.issubset(set( self.value_names() )) ):
                print("No data to store.")
                return
            else:
                path = self.GEFdata

                #Create a dictionary used to initialise the pandas DataFrame
                dic = {}

                #remember the original units of the GEF
                units=self.get_units()

                #Data is always stored unitless
                self.set_units(False)

                for key in storeables:
                    dic[key] = getattr(self, key).value
                
                #Create pandas data frame and store the dictionary under the user-specified path
                output_df = pd.DataFrame(dic)  
                output_df.to_csv(path)

                #after storing data, restore original units
                self.set_units(units)
        return



def GEF(modelname, settings):
    @_model_setup(modelname, settings)
    class GEF(BaseGEF):
        pass

    return GEF


def print_summary(sol):
    print("GEF run completed with the following statistics")
    for attr in sol.keys():
        if attr not in ["y", "t", "y_events", "t_events", "sol", "events"]:
            print(rf" - {attr} : {getattr(sol, attr)}")
    events = sol.events
    if np.array([(len(event["t"])==0) for event in events.values()]).all():
        print("No events occured during the run")
    else:
        print("The following events occured during the run:")
        for event in events.keys():
            time = events[event]["t"]
            efold = events[event]["N"]
            if len(time > 0):
                print(f"  - {event} at t={time} or N={efold}")
    return