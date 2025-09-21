import pandas as pd
import numpy as np
from ._docs import generate_docs, docs_gef
from .bgtypes import BGSystem, Val, Func
from .models import classic

import importlib
import os

from numbers import Number
from types import NoneType


class BaseGEF(BGSystem):
    

    GEFSolver = classic.solver
    """The solver used to solve the GEF equations in `run`."""
    ModeSolver = classic.MbM
    """The mode solver used for mode-by-mode cross checks."""
    _input_signature = classic.input
    define_units = staticmethod(classic.define_units)

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
            file to path for GEF data
        MbMdata : None or str:
            file to path for mode data

        Raises
        ------
        KeyError
            if a necessary input is missing.
        TypeError
            if the input is of the wrong type.
        """    
        user_input = {"constants":consts, "initial data":init_data, "functions":funcs}

        #Check that all necessary input is present and that its data type is correct
        for input_type, input_dict  in user_input.items():
            self._check_input(input_dict, input_type)

        omega, mu = self.define_units(*user_input.values())

        known_objects = set().union(*[item for key, item in self.GEFSolver.known_variables.items() if key!="gauge"] )

        super().__init__(known_objects, omega, mu)

        #Add initial data to BGSystem
        for name, constant in user_input["constants"].items():
            self.initialise(name)(constant)
        for name, function in user_input["functions"].items():
            self.initialise(name)(function)
        for name, value in user_input["initial data"].items():
            self.initialise(name)(value)

        #initialise the other values with dummy variables.
        for obj in self.quantity_set():
            if obj.name not in (self.variable_names() + self.function_names() + self.constant_names()):
                if issubclass(obj, Val):
                    self.initialise(obj.name)(0)
                elif issubclass(obj, Func):
                    self.initialise(obj.name)(lambda *x: 0)

        #Add information about file paths
        self.GEFdata = GEFdata
        """Path to GEF data used by `load_GEFdata` and `save_GEFdata`."""
        self.MbMdata = MbMdata
        """Path to mode data. Load using `.mbm.GaugeSpec.read_spec`"""

        self._completed=False

    @classmethod
    def print_input(cls):
        """
        Print the input required to initialize the class.
        """
        print("This GEF model requires the following input:")
        for key, item in cls._input_signature.items():
            print(f"\t {key.capitalize()}: {item}")
        return

    def _check_input(self, input_data : dict, input_type : str):
        for key in self._input_signature[input_type]:
            try:
                assert key in input_data.keys()
            except AssertionError:
                raise KeyError(f"Missing input in '{input_type}': '{key}'")

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
    
    def run(self, ntr:int=150, tend:float=120, nmodes:int=500, mbm_attempts:int=5,  resume_mbm:bool=True,
              err_tol:float=0.1, err_thr:float=0.025, binning:int=5, integrator:str="simpson", print_stats:bool=True, **solver_kwargs):
        """
        Solve the ODE's of the GEF using `GEFSolver`. Cross check the solution using `ModeSolver`.

        The `GEFSolver` is initialized using the initial conditions defined by the class.
        If the solver returns a successful solution, `ModeSolver.compute_spectrum` computes a gauge field spectrum
         to perform a mode-by-mode cross check (unless `nmodes=None`).
        If the mode-by-mode cross check is a success, the solution is stored in the underlying `BGSystem` of the class.
        Otherwise, the `GEFSolver` tries to self correct using the gauge field spectrum. This is attempted for `mbm_attempts` or until successful.

        Parameters
        ----------
        ntr : int
            initial truncation number `GEFSolver.ntr`
        tend : float
            initial target time for `GEFSolver.tend`
        nmodes : int or None
            The number of modes computed by `ModeSolver`. If None, no cross-check is performed.
        mbm_attempts : int
            number of mode-by-mode self correction attempts
        resume_mbm : bool
            If `True` use `ModeSolver.update_spectrum` in case multiple mode-by-mode comparisons are needed.
        err_tol : float
            Passed to `mbm_crosscheck`.
        err_thr : float
            Passed to `mbm_crosscheck`.
        binning : int
            Passed to `mbm_crosscheck`.
        integrator : str
            integrator for `mbm_crosscheck` ('simpson' is advised)
        print_stats : bool
            If `True`, a summary report is printed for the returned solution.
        solver_kwargs
            he `settings` of `GEFSolver` (see `GEFSolver.settings`)

        Returns
        -------
        sol
            the result of `GEFSolver.compute_GEF_solution`
        spec : GaugeSpec or None
            the result of `ModeSolver.compute_spectrum`

        Raises
        ------
        RuntimeError
            if no successful solution was obtained.
        """
        if self._completed:
            print("GEF data already computed.")
            return None, None
        solver = self.GEFSolver(self)

        #Configuring GEFSolver
        solver.ntr=ntr
        solver.tend=tend
        solver.update_settings(**solver_kwargs)


        integrator_kwargs = {"integrator":integrator, "epsabs":solver.settings["atol"], "epsrel":solver.settings["rtol"]}

        done=False
        vals = BGSystem.from_system(self, copy=True)
        sol = None
        spec = None
        t_reinit = 0.
        attempt=0

        while not(done) and attempt<mbm_attempts:
            attempt +=1
            #This can be taken care of internally. The GEF should not need to get sol objects...
            sol_new = solver.compute_GEF_solution()
            sol = self._update_sol(sol, sol_new)
            solver.parse_arr_to_sys(sol.t, sol.y, vals)

            if nmodes is not None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeSolver(vals)

                rtol = solver.settings["rtol"]

                if resume_mbm and attempt > 1:
                    spec = MbM.update_spectrum(spec, t_reinit, rtol=rtol)
                else:
                    spec = MbM.compute_spectrum(nmodes, rtol=rtol)
                print("Performing mode-by-mode comparison with GEF results.")

                agreement, reinit_spec = self.mbm_crosscheck(spec, vals, err_tol= err_tol, err_thr=err_thr, binning=binning,
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
                self.set_units(False)
                for obj in self.variable_list():
                    obj.set_value(getattr(vals, obj.name).value)
                self.set_units(True)
                self._completed = True
            else:
                print("The run terminated on with an error, check output for details.")

            
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
        if np.array([(len(event)==0) for event in events.values()]).all():
            print("No events occured during the run")
        else:
            print("The following events occured during the run:")
            for event in events.keys():
                time = events[event]
                if len(time > 0):
                    print(f"  - {event} at t={time}")
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
    def mbm_crosscheck(spec, vals:BGSystem, err_tol:float, err_thr:float, binning:int, **integrator_kwargs):
        """
        Estimate the error of a GEF solution using `.mbm.GaugeSpec.estimate_GEF_error`.

        If either the RMS error or the final error exceeds `err_tol`, the solution is rejected.
        
        Parameters
        ----------
        vals : BGSystem
            contains the GEF solution.
        err_tol : float
            the tolerance on the RMS and final error.
        err_thr : float
            passed to `estimate_GEF_error`.
        binning : int
            passed to `estimate_GEF_error`.
        integratorkwargs:
            passed to kwargs of `estimate_GEF_error`.

        Returns
        -------
        agreement : bool
            indicates if the solution is accepted or rejected.
        reinit_slice : SpecSlice
            the spectrum with which the GEF solver is re-initialized.
        """
        errs, terr, _ = spec.estimate_GEF_error(vals, err_thr=err_thr, binning=binning, **integrator_kwargs)

        reinit_inds = []
        agreement=True
        for err in errs:
            rmserr = np.sqrt(np.sum(err**2)/len(err))
            if max(err[-1], rmserr) > err_tol:
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

        Note, data is always loaded assuming numerical units.

        Parameters
        ----------
        path : None or str
            If None, loads data from `GEFdata`. Otherwise, loads data from the specified path.

        Raises
        ------
        Exception
            if `path` is None but `GEFdata` is also None.
        FileNotFoundError
            if no file is found at `path`.
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
                raise AttributeError(f"The data table you tried to load contains an unknown quantity: '{key}'")
        
        #Store current units to switch back to later
        units=self.get_units()

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.set_units(False)
        #Load data into background-value attributes
        for key, values in data.items():
            self.initialise(key)(values)

        self.set_units(units)
        self._completed=True

        return

    def save_GEFdata(self, path : NoneType|str=None):
        """
        Save the data in the current GEF instance in an output file.

        Note, data is always stored in numerical units.

        Parameters
        ----------
        path : str
            If None, stores data in `GEFdata`. Else, stores data in the specified file.

        Raises
        ------
        Exception
            if 'path' is None but `GEFdata` is also None.
        
        """
        if path is None:
            path=self.GEFdata
        else:
            self.GEFdata=path
        
        if path is None:
            raise Exception("You did not specify the file under which to store the GEF data. Set 'GEFdata' to the location where you want to save your data.")

        else:
            storeables = self.variable_names     
            #Check that all dynamic and derived quantities are initialised in this GEF instance
            if len(storeables)==0:
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



def _load_model(model : str, user_settings : dict):
    """
    Import and execute a module defining a GEF model.

    Parameters
    ----------
    model : str
        The name of the GEF model or a full dotted import path (e.g., "path.to.module").
    settings : dict
        A dictionary containing updated settings for the module.

    Returns
    -------
    ModuleType
        The configured module.
    """

    # Case 1: Bare name, resolve to ./models/{name}.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"models/{model}.py")

    if os.path.exists(modelpath):
        spec = importlib.util.spec_from_file_location(model, modelpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    else:
        # Case 2: Try treating it as a dotted import path
        try:
            mod = importlib.import_module(model)
        except ImportError as e:
            raise FileNotFoundError(
                f"No model file found at '{modelpath}' and failed to import '{model}'"
                ) from e
        
    if hasattr(mod, "settings") and isinstance(user_settings, dict):
        for key, item in user_settings.items():
            if key in mod.settings:
                mod.settings[key] = item
                print(f"Updating '{key}' to '{item}'.")
            else:
                print(f"Ignoring unknown model setting '{key}'.")
        mod.interpret_settings()

    return mod


def GEF(modelname:str, settings:dict):
    """
    Define a custom subclass of BaseGEF adapted to a new GEF model.

    Parameters
    ----------
    modelname : str
        The name of the GEF model or a full dotted import path (e.g., "path.to.module").
    settings : dict
        a dictionary of settings used by the model

    Returns
    -------
    CustomGEF
        a custom subclass of BaseGEF
        
    """
    model = _load_model(modelname, settings)
    class CustomGEF(BaseGEF):
        GEFSolver = model.solver
        """The solver used to solve the GEF equations in `run`."""
        ModeSolver = model.MbM
        """The mode solver used for mode-by-mode cross checks."""
        _input_signature = model.input
        define_units = staticmethod(classic.define_units)

    CustomGEF.__qualname__ = model.name
    CustomGEF.__module__ = __name__

    return CustomGEF

generate_docs(docs_gef.DOCS)