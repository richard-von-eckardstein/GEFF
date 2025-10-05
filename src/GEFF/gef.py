import pandas as pd
import numpy as np
import inspect
from ._docs import generate_docs, docs_gef
from .bgtypes import BGSystem, Val, Func
from .models import load_model, classic

from numbers import Number
from typing import Callable

class BaseGEF(BGSystem):
    
    GEFSolver = classic.solver
    """The solver used to solve the GEF equations in `run`."""
    ModeSolver = classic.MbM
    """The mode solver used for mode-by-mode cross checks."""
    _input_signature = classic.input_dic
    define_units = staticmethod(classic.define_units)

    def __init__(self, **kwargs):
        """
        Initialize the GEF from user-specified initial data, constants and functions.

        Parameters
        ----------
        kwargs 
            To access a list of kwargs for this model use `print_kwargs`.

        Raises
        ------
        KeyError
            if a necessary input is missing.
        TypeError
            if the input is of the wrong type.
        """

        for val in self._input_signature:
            if val.name not in kwargs.keys():
                raise KeyError(f"Missing input '{val.name}'")
            else:
                key = val.name
                item = kwargs[key]
            if issubclass(val, Val) and not(isinstance(item, Number)):
                raise TypeError(f"Input for '{key}' should be 'Number' type.")
            elif issubclass(val, Func) and not(callable(item)):
                raise TypeError(f"Input for '{key}' must be callable.")

        #check which keys are needed for define_units
        expected_keys = inspect.getfullargspec(self.define_units).args
        omega, mu = self.define_units(**{key:kwargs[key] for key in expected_keys})

        known_objects = set().union(*[item for key, item in self.GEFSolver.known_variables.items() if key!="gauge"] )

        super().__init__(known_objects, omega, mu)

        for key, item in kwargs.items():
            self.initialise(key)(item)

        #initialise the other values with dummy variables.
        for obj in self.quantity_set():
            if not(hasattr(self, obj.name)):
                if issubclass(obj, Val):
                    self.initialise(obj.name)(0)
                elif issubclass(obj, Func):
                    self.initialise(obj.name)(lambda *x: 0)    

        self._completed = False

    @classmethod
    def print_kwargs(cls):
        """
        Print the input required to initialize the class.
        """
        print("This GEF model expects the following input:\n")
        for i in cls._input_signature:
            print(f" * {i.get_description()}")
        print()
        return
    
    @classmethod
    def print_known_variables(cls):
        """
        Print a list of known variables for this model
        """
        print("This GEF model knows the following variables:\n")
        for key, item in cls.GEFSolver.known_variables.items():
            if len(item) > 0:
                print(f"{key}:")
                for i in item:
                    print(f" * {i.get_description()}")
        print()
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
                          t={reinit_spec['t']:.1f}, N={reinit_spec['N']:.1f}.\n")

                    solver.set_initial_conditions_to_MbM(sol, reinit_spec)
                
            else:
                done=True
        
        if done:
            if print_stats:
                self._print_summary(sol)
            if sol.success:
                print("\nStoring results in GEF instance.")
                self.units = False
                for obj in self.variable_list():
                    obj.value = (getattr(vals, obj.name).value)
                self.units = True
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
            print("No events occurred during the run")
        else:
            print("The following events occurred during the run:")
            for event, time in events.items():
                if len(time) > 0:
                    if len(time) > 1:
                        tstr = [f"{t:.1f}" for t in time]
                    else:
                        tstr = f"{time[0]:.1f}"
                    print(f"  - {event} at t={tstr}")
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
            ind_overlap = np.searchsorted(sol_old.t, sol_new.t[0], "right")
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
        
    def initialize_modesolver(self):
        """
        Return a `ModeSolver` instance for the GEF

        Returns
        -------
        mbm : ModeSolver
            a mode solver instance initialized with the GEF solution.

        Raises
        ------
        ValueError
            if no solution is contained in the GEF.
        """
        if self._completed:
            return self.ModeSolver(self)
        else:
            raise ValueError("No GEF solution for which the ModeSolver can be initialized.")
        
    def mbm_crosscheck(self, spec, vals:BGSystem, err_tol:float, err_thr:float, binning:int, **integrator_kwargs):
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
        GFs = self.GEFSolver.known_variables["gauge"]
        agreement=True

        for GF in GFs:
            ref = [a.name for a in GF.associated]
            cut = GF.cutoff.name

            errs, terr, _ = spec.estimate_GEF_error(vals, references=ref, cutoff=cut,
                                                     err_thr=err_thr, binning=binning, **integrator_kwargs)

            reinit_inds = []
            
            for err in errs:
                rmserr = np.sqrt(np.sum(err**2)/len(err))
                if max(err[-1], rmserr) > err_tol:
                    agreement=False
                    #find where the error is above 5%, take the earliest occurrence, reduce by 1
                    err_ind = np.where(err > err_thr)[0][0]-1             
                else:
                    err_ind = len(terr)-1
                reinit_inds.append( err_ind )

            t0 = terr[min(reinit_inds)]

            ind = np.searchsorted(spec["t"],t0, "left")

            reinit_slice = spec.tslice(ind)

        return agreement, reinit_slice

    def load_GEFdata(self, path : str):
        """
        Load data and store its results in the current GEF instance.

        Note, data is always loaded assuming numerical units.

        Parameters
        ----------
        path : None or str
            Path to load data from

        Raises
        ------
        FileNotFoundError
            if no file is found at `path`.
        AttributeError
            if the file contains a column labeled by a key which does not match any GEF-value name.
        """
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
        og_units=self.units

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.units = False
        #Load data into background-value attributes
        for key, values in data.items():
            self.initialise(key)(values)

        self.units = og_units
        self._completed=True

        return

    def save_GEFdata(self, path : str):
        """
        Save the data in the current GEF instance in an output file.

        Note, data is always stored in numerical units.

        Parameters
        ----------
        path : str
            Path to store data in.

        Raises
        ------
        ValueError
            if the GEF object has no data to store.
        """

        storeables = self.variable_names     
        #Check that all dynamic and derived quantities are initialised in this GEF instance
        if not(self._completed):
            raise ValueError("No data to store.")
        
        #Create a dictionary used to initialise the pandas DataFrame
        dic = {}

        #remember the original units of the GEF
        og_units=self.units

        #Data is always stored unitless
        self.units = False

        for key in storeables:
            dic[key] = getattr(self, key).value
        
        #Create pandas data frame and store the dictionary under the user-specified path
        output_df = pd.DataFrame(dic)  
        output_df.to_csv(path)

        #after storing data, restore original units
        self.units = og_units
        return


def GEF(modelname:str, settings:dict={}):
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
    model = load_model(modelname, settings)
    signature  = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)]
    for val in model.input_dic:
        if issubclass(val, Func):
            annotation = Callable
        else:
            annotation = float
        signature.append(inspect.Parameter(val.name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation))
    
    class CustomGEF(BaseGEF):
        GEFSolver = model.solver
        """The solver used to solve the GEF equations in `run`."""
        ModeSolver = model.MbM
        """The mode solver used for mode-by-mode cross checks."""
        _input_signature = model.input_dic
        define_units = staticmethod(model.define_units)

    CustomGEF.__init__.__signature__ = inspect.Signature(signature)

    CustomGEF.__qualname__ = model.name
    CustomGEF.__module__ = __name__

    return CustomGEF

generate_docs(docs_gef.DOCS)