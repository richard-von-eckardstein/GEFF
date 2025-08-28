from ._docs import gef_docs, generate_docs
import pandas as pd
import numpy as np
from .bgtypes import BGSystem

import importlib.util as util
import os

from .solver import BaseGEFSolver
from .mode_by_mode import BaseModeSolver
from .models import classic
from numbers import Number
from types import NoneType
from typing import ClassVar

class MissingInputError(Exception):
    pass

def _load_model(name : str, user_settings : dict={}):
    """
    Import and execute a module defining a GEF model.

    Parameters
    ----------
    modelname : str
        the name of the GEF model
    settings : dict
        a dictionary containing updated settings for the module

    Returns
    -------
    ModuleType
        the executed module
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"models/{name}.py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(name, modelpath)
        mod  = util.module_from_spec(spec)
        #update the settings according to the user input
        for key, item in user_settings.items():
            try:
                mod.settings[key] = item
            except AttributeError:
                print(f"Ignoring unknown model setting '{key}'.")
        #execute the module.
        spec.loader.exec_module(mod)
        return mod
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found under '{modelpath}'")
    

def _add_model_specifications(model_name, user_settings):
    """
    Define a GEF subclass based on a module.

    Parameters
    ----------
    modelname : str
        the name of the GEF model
    settings : dict
        a dictionary containing updated settings for the module

    Returns
    -------
    ModuleType
        the executed module
    """
    def GEF_decorator(cls):
        #load the model file
        model = _load_model(model_name, user_settings)

        #import information on input and how to handle it
        cls._input_signature = model.input
        cls._input_handler = staticmethod(model.define_units)

        #import information for solver:
        cls.GEFSolver = model.solver
        #import Mode-By-Mode class:
        cls.ModeSolver = model.MbM

        cls._object_classification = { key:{i.name for i in item} for key, item in model.solver.known_variables.items()}

        return cls
    return GEF_decorator


class BaseGEF(BGSystem):
    """
    This class is the primary interface of a GEF model.

    The class contains a `GEFSolver` and a `ModeSolver` that are used to solve GEF equations in `run`.

    The class also stores the evolution of the GEF-model variables as obtained from `run` or `load_GEFdata`.
    You can access these variables like a `GEFF.bgtypes.BGSystem`. E.g., you can access the $e$-folds variable through the attribute `N`.

    As a child of `BGSystem`, the GEFF can be passed to other tools initialised by `BGSystem`'s. For example, to compute the tensor power spectrum from your GEF solution,
    you can pass it to `GEFF.tools.pt.PT`.

    The `BaseGEF` contains the model `GEFF.models.classic`. To create a custom GEF model from a model file, use the class factory `GEF`.
    """

    GEFSolver : ClassVar[BaseGEFSolver] = classic.solver
    """
    The solver used to compute the GEF evolution.
    """
    ModeSolver : ClassVar[BaseModeSolver] = classic.MbM
    """
    The mode solver used for mode-by-mode cross checks.
    """
    
    _input_signature = classic.input
    define_units = staticmethod(classic.define_units)
    _object_classification = { key:{i.name for i in item} for key, item in classic.quantities.items()}

    def __init__(
                self, consts : dict, init_data : dict, funcs : dict, 
                GEFdata: NoneType|str = None, MbMdata: NoneType|str = None
                ):
        """
        Define values for input constants, initial data, and functions used by the GEF.

        This also initializes the underlying `BGSystem` class using `define_units`.

        Parameters
        ----------
        consts : dict
            user specified values for constants
        init_data : dict
            user specified initial data for the GEF
        funcs : dict
            user specified functions of GEF variables
        GEFdata : None or str:
            file to path where to load and save GEF data
        MbMdata : None or str:
            file to path where to load and save mode by mode data

        """    
        user_input = {"constants":consts, "initial data":init_data, "functions":funcs}

        #Check that all necessary input is present and that its data type is correct
        for input_type, input_dict  in user_input.items():
            self._check_input(input_dict, input_type)

        H0, MP = self.define_units(user_input)

        super().__init__(self._known_objects, H0, MP)

        #Add initial data to BGSystem
        for name, constant in user_input["constants"].items():
            self.initialise(name)(constant)
        for name, function in user_input["functions"].items():
            self.initialise(name)(function)
        for name, value in user_input["initial data"].items():
            self.initialise(name)(value)

        #initialise the other values with dummy variables.
        for name in self.quantity_names():
            if name not in (self.value_names() + self.function_names()):
                self.initialise(name)(0)

        #Add information about file paths
        self.GEFdata = GEFdata
        """Path to GEF data used by `load_GEFdata` and `save_GEFdata`."""
        self.MbMdata = MbMdata
        """Path to mode data. Load using `GEFF.mode_by_mode.GaugeSpec.read_spec`"""

    @classmethod
    def print_input(cls):
        print("This GEF model requires the following input:")
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
        self.set_units(False)

        solver = self.GEFSolver(self)

        #Configuring GEFSolver
        solver.ntr=ntr
        solver.tend=tend
        solver_kwargs = {setting : kwargs[setting] for setting in solver.settings if setting in kwargs}        
        solver.update_settings(**solver_kwargs)

        #Configuring ModeSolver
        MbMattempts = kwargs.get("MbMattempts", 5)
        binning = kwargs.get("binning", 5)
        err_thr = kwargs.get("err_thr", 0.025)
        resumeMbM = kwargs.get("resumeMbM", True)
        int_method = kwargs.get("integrator", "simpson")

        integrator_kwargs = {"integrator":int_method, "epsabs":solver.settings["atol"], "epsrel":solver.settings["rtol"]}

        done=False
        vals = BGSystem.from_system(self, copy=True)
        sol = None
        spec = None
        t_reinit = 0.
        attempt=0

        while not(done) and attempt<MbMattempts:
            attempt +=1
            #This can be taken care of internally. The GEF should not need to get sol objects...
            sol_new = solver.compute_GEF_solution()
            sol = self._update_sol(sol, sol_new)
            solver.parse_arr_to_sys(sol.t, sol.y, vals)

            if nmodes is not None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeSolver(vals)

                rtol = solver.settings["rtol"]

                if resumeMbM and attempt > 1:
                    spec = MbM.update_spectrum(spec, t_reinit, rtol=rtol)
                else:
                    spec = MbM.compute_spectrum(nmodes, rtol=rtol)
                print("Performing mode-by-mode comparison with GEF results.")

                agreement, reinit_spec = self.MbMcrosscheck(spec, vals, err_thr=err_thr, binning=binning,
                                                             **integrator_kwargs)

                if agreement:
                    print("The mode-by-mode comparison indicates a convergent GEF run.\n")
                    done=True
                
                else:
                    t_reinit = reinit_spec["t"]
                    print(f"Attempting to solve GEF using self-correction starting from \
                          t={np.round(reinit_spec['t'], 1)}, N={np.round(reinit_spec['N'], 1)}.\n")

                    solver.set_initial_conditions_to_MbM(sol, reinit_spec)
                
            else:
                done=True
        
        if done:
            
            if print_stats:
                self._print_summary(sol)
            if sol.success:
                print("\nStoring results in GEF instance.")
                for obj in self.value_list():
                    obj.set_value(getattr(vals, obj.name).value)
            else:
                print("The run terminated on with an error, check output for details.")

            self.set_units(True)
            return sol, spec
        
        else:
            raise RuntimeError(f"GEF did not complete after {attempt} attempts.")
    
    @staticmethod
    def _print_summary(sol):
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
        
    @staticmethod
    def _update_sol(sol_old, sol_new):
        """
        Update an old GEF solution with a new one, overwriting the overlap.
        """
        if sol_old is None:
            return sol_new
        else:
            sol = sol_old
            ind_overlap = np.where(sol_new.t[0] >= sol_old.t)[0][-1]
            sol.t = np.concatenate([sol_old.t[:ind_overlap], sol_new.t])

            if sol_old.y.shape[0] < sol_new.y.shape[0]:
                #if ntr increased from one solution to the next, fill up sol_old with zeros to match sol_new
                fillshape = (sol_new.y.shape[0] - sol_old.y.shape[0], sol_old.y.shape[1])
                yfill = np.zeros( fillshape )
                sol_old.y = np.concatenate([sol_old.y, yfill], axis=0)

            sol.y = np.concatenate([sol_old.y[:,:ind_overlap], sol_new.y], axis=1)
            sol.events.update(sol_new.events)
            for attr in ["nfev", "njev", "nlu"]:
                setattr(sol, attr, getattr(sol_old, attr) + getattr(sol_new, attr))
            for attr in ["message", "success", "status"]:
                setattr(sol, attr, getattr(sol_new, attr))
            return sol
        
     #move to GEF
    @staticmethod
    def MbMcrosscheck(spec, vals, err_thr, binning, **MbMkwargs):
        errs, terr, _ = spec.estimate_GEF_error(vals, err_thr=err_thr, binning=binning, **MbMkwargs)

        reinit_inds = []
        agreement=True
        for err in errs:
            rmserr = np.sqrt(np.sum(err**2)/len(err))
            if max(err[-1], rmserr) > 0.10:
                agreement=False
                #find where the error is above 5%, take the earliest occurrence, reduce by 1
                inds = np.where(err > err_thr)
                err_ind = inds[0][0]-1               
            else:
                err_ind = len(terr)-1
            reinit_inds.append( err_ind )

        t0 = terr[min(reinit_inds)]

        ind = np.where(spec["t"] <= t0)[0][-1]

        reinit_slice = spec.tslice(ind)

        return agreement, reinit_slice

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
        names = self.quantity_names()
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
            if None, stores data in self.GEFdata, else, stores data in the specified file.

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
    """somedoc"""
    @_add_model_specifications(modelname, settings)
    class GEF(BaseGEF):
        pass

    return GEF


#generate_docs(gef_docs.DOCS)