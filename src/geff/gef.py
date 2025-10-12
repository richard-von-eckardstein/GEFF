import numpy as np
import inspect
from ._docs import generate_docs, docs_gef
from .bgtypes import BGSystem, Val, Func
from .models import load_model, pai

from numbers import Number
from typing import Callable

class BaseGEF:
    
    GEFSolver = pai.solver
    """The solver used to solve the GEF equations in `run`."""
    ModeSolver = pai.MbM
    """The mode solver used for mode-by-mode cross checks."""
    _input_signature = pai.model_input
    define_units = staticmethod(pai.define_units)

    def __init__(self, **kwargs):
        """
        Initialize the GEF from user-specified initial data, constants and functions.

        Parameters
        ----------
        kwargs 
            To access a list of kwargs for this model use `print_input`.

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

        initial_data = BGSystem(known_objects, omega, mu)

        for key, item in kwargs.items():
            initial_data.initialise(key)(item)

        #initialise the other values with dummy variables.
        for obj in initial_data.quantity_set():
            if not(hasattr(initial_data, obj.name)):
                if issubclass(obj, Val):
                    initial_data.initialise(obj.name)(0)
                elif issubclass(obj, Func):
                    initial_data.initialise(obj.name)(lambda *x: 0)    

        self.initial_data = initial_data

        # test if ODE solver works
        try:
            self.GEFSolver(initial_data)
        except Exception:
            raise ModelError("Initializing the GEFSolver returned an error.")


    @classmethod
    def print_input(cls):
        """
        Print the input required to initialize the class.
        """
        print("This GEF model expects the following input:\n")
        for i in cls._input_signature:
            print(f" * {i.get_description()}")
        print()
        return
    
    @classmethod
    def pring_ingredients(cls):
        """
        Print a list of known variables, functions and constants for this model
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
            the number of modes computed by `ModeSolver`. If None, no cross-check is performed.
        mbm_attempts : int
            number of mode-by-mode self correction attempts
        resume_mbm : bool
            if `True` use `ModeSolver.update_spectrum` in case successive mode-by-mode comparisons are needed.
        err_tol : float
            passed to `mbm_crosscheck`.
        err_thr : float
            passed to `mbm_crosscheck`.
        binning : int
            passed to `mbm_crosscheck`.
        integrator : str
            integrator for `mbm_crosscheck` ('simpson' is advised)
        print_stats : bool
            if `True`, a summary report is printed for the returned solution.
        solver_kwargs
            the `settings` of `GEFSolver` (see `GEFSolver.settings`)

        Returns
        -------
        vals : BGSystem
            contains the background evolution obtained from `GEFSolver`
        spec : GaugeSpec or None
            the result of `ModeSolver.compute_spectrum`
        sol
            the full result of `GEFSolver.compute_GEF_solution`
        

        Raises
        ------
        RuntimeError
            if no successful solution was obtained.
        """
        solver = self.GEFSolver(self.initial_data)

        #Configuring GEFSolver
        solver.ntr=ntr
        solver.tend=tend
        solver.update_settings(**solver_kwargs)

        integrator_kwargs = {"integrator":integrator, "epsabs":solver.settings["atol"], "epsrel":solver.settings["rtol"]}

        done=False
        vals = BGSystem.from_system(self.initial_data, copy=True)
        vals.units = False
        sol = None
        spec = None
        t_reinit = 0.
        attempt=0

        print()

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
                    print(f"Attempting to solve GEF using self-correction starting from t={reinit_spec['t']:.1f}, N={reinit_spec['N']:.1f}.\n")

                    solver.set_initial_conditions_to_MbM(sol, reinit_spec)
                
            else:
                done=True
        
        if done:
            if print_stats:
                self._print_summary(sol)
            if sol.success:
                vals.units = True
                return vals, spec, sol
            else:
                print("The run terminated on with an error, check output for details.")
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
            print("No events occurred during the final run.")
        else:
            print("The following events occurred during the final run:")
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
            ind_overlap = np.searchsorted(sol_old.t, sol_new.t[0], "left")
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
            ref = [a.name for a in GF.zeros]
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
        Load data and return a BGSystem with the results.

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
        newsys = BGSystem.from_system(self.initial_data, copy=True)
        
        newsys.load_variables(path)

        return newsys

def compile_model(modelname:str, settings:dict={}):
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
    for val in model.model_input:
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
        _input_signature = model.model_input
        define_units = staticmethod(model.define_units)

    CustomGEF.__init__.__signature__ = inspect.Signature(signature)

    CustomGEF.__qualname__ = model.name
    CustomGEF.__module__ = __name__

    return CustomGEF

class ModelError(Exception):
    """
    Exception for errors when compiling a GEF model.
    """
    pass

generate_docs(docs_gef.DOCS)