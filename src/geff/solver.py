
import numpy as np
from .bgtypes import BGSystem, t, N, a, H
from .mbm import SpecSlice
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, ClassVar
from ._docs import generate_docs, docs_solver
from .utility.general import AuxTol

class BaseGEFSolver:
    known_variables : ClassVar[dict] = {"time":[t], "dynamical":[N], "static":[a], "constant":[H], "function":[], "gauge":[]}
    """
    Classifies variables used by the solver according to:
    * 'time': the name of the time parameter of the ODE's (should be "t") 
    * 'dynamical': variables evolved by the EoM (not 'gauge')
    * 'gauge': tower of gauge-field expectation values evolved by the EoM
    * 'static': variables computed from 'dynamical' and 'gauge'
    * 'constant': constant variables
    * 'function': functions of the above variables.
    """

    known_events : ClassVar[dict] = {}
    """The `Event` objects which are tracked by the solver."""

    def __init__(self, init_sys : BGSystem):
        """
        Pass initial data to the solver.

        Parameters
        ----------
        init_sys : BGSystem
            initial data used by the solver
        """

        self.init_vals : BGSystem = BGSystem.from_system(init_sys, copy=True)
        """Initial data for the EoM's defined at $t_0 = 0$."""

        self.init_vals.units = False

        self.settings : dict ={"atol":1e-20, "rtol":1e-6, "attempts":5, "solvermethod":"RK45", "ntrstep":10}
        """
        A dictionary of internal settings used by the class:
        - atol: absolute tolerance used by `solve_ivp` (default: 1e-20)
        - rtol: relative tolerance used by `solve_ivp` (default: 1e-6)
        - solvermethod: method used by `solve_ivp` (default: 'RK45')
        - attempts: attempts made by `compute_GEF_solution` (default: 5)
        - ntrstep: `ntr` increment used in `compute_GEF_solution` (default: 10)
        """

        self.ntr : int = 100
        r"""Truncation number $n_{\rm tr}$, for truncated towers of ODE's."""

        self.tend : float = 120.
        r"""The time $t_{\rm end}$ up to which the EoMs are solved."""

        #initial conditions on initialisation
        self.set_initial_conditions_to_default()

        self._test_ode()

    @classmethod
    def toggle_event(self, event_name : str, toggle : bool):
        """
        Disable or enable an `Event` for the solver.

        Parameters
        ----------
        event_name : str
            the name of the target
        toggle : bool
            if the event should be active or inactive
        """
        if event_name in [event for event in self.known_events.keys()]:
            self.known_events[event_name].active = toggle
            humanreadable = {True:"active", False:"inactive"}
            print(f"The event '{event_name}' is now {humanreadable[toggle]}")
        else:
            print(f"Unknown event: '{event_name}'")
        return

    def compute_GEF_solution(self):
        """
        An algorithm to solve the GEF equations.

        The solver attempts to solve the EoMs several times using `solve_eom`. 
        If this returns a `TruncationError`, `ntr` is increased by `settings['ntrstep']`.
        Afterwards, `solve_eom` is called again until it returns a result.
        This is done for `settings['attempts']` times or until `ntr=200` is reached.

        Returns
        -------
        sol
            a bunch object returned by `solve_ivp` containing the solution

        Raises
        ------
        RuntimeError
             if no solution was obtained after the maximum number of attempts.
        """
        maxntr = 200
        maxattempts = self.settings["attempts"]
        ntrstep = self.settings["ntrstep"]

        attempt=0
        done = False
        sol = None
        #Run GEF
        while not(done) and (attempt<maxattempts):
            attempt+=1
            try:
                #Try to get convergent solution
                t0, yini, vals = self.initial_conditions()
                sol = self.solve_eom(t0, yini, vals)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except TruncationError:
                if self.ntr<maxntr:
                    self._increase_ntr( min(ntrstep, maxntr-self.ntr) )
                else:
                    print("Cannot increase ntr further.")
                    break
            else:
                done=True
        
        if sol is None:
            raise RuntimeError(f"No GEF solution after {attempt} attempts.")
        else:
            if sol.success:
                print("Successful GEF solution obtained. Proceeding.\n")
            else:
                print("Processing unsuccessful solution.\n")
        
        return sol
    
    ### handle initial conditions ###

    @staticmethod
    def vals_to_yini(vals : BGSystem, ntr : int) -> np.ndarray:
        """
        Create an array of initial data from a BGSystem.

        Parameters
        ----------
        vals : BGSystem
            a unit system with initial data
        ntr : int
            truncation number (relevant for dynamical gauge-fields)

        Returns
        -------
        yini : NDarray
            an array of initial data
        """
        vals.units = False #ensure the system is in numerical units

        #In the simple version, ntr has no meaning, as there are no gauge fields
        yini = np.zeros((1)) 
        #get N from vals
        yini[0] = vals.N.value 
        return yini

    def set_initial_conditions_to_default(self):
        """
        Configure the solver to use initial data from `init_vals` using `vals_to_yini`.
        """
        def default_initial_conditions():
            """
            Compute initial data with `init_vals` using `vals_to_yini`.
            """
            t0 = 0
            vals = BGSystem.from_system(self.init_vals, True)
            vals.units = False
            yini = self.vals_to_yini(vals, self.ntr)
            return t0, yini, vals
        self.initial_conditions = staticmethod(default_initial_conditions)
        return
        
    
    def set_initial_conditions_to_MbM(self, sol, reinit_spec : SpecSlice):
        r"""
        Configure the solver to initialise data from a mode-by-mode solution.

        Used for mode-by-mode self correction. 
        Gauge-bilinears, $F_{\mathcal X}^{(n>1)}$ are re-initialized using `.mbm.SpecSlice.integrate_slice`.
        
        Parameters
        ----------
        sol
            a bunch object returned by `solve_ivp` containing the solution
        reinit_spec : SpecSlice
            spectrum at time of reinitialization
        """
        def MbM_initial_conditions():
            ntr = self.ntr

            t_reinit = reinit_spec["t"]

            reinit_ind = np.searchsorted(sol.t,t_reinit, "left")

            #Create unit system (copy to also get constants and functions):
            temp = BGSystem.from_system(self.init_vals, copy=True)

            #get number of regular dynamical variables
            n_dynam = len(self.known_variables["dynamical"])

            #construct yini
            yini = np.zeros(((ntr+1)*3+n_dynam))

            #parse everyting except for GEF-bilinears with n>1 to yini
            yini[:n_dynam+3] = sol.y[:n_dynam+3,reinit_ind]

            # compute En, Bn, Gn, for n>1 from Modes
            yini[n_dynam+3:] = np.array([reinit_spec.integrate_slice(n=n, integrator="simpson")
                                            for n in range(1,ntr+1)])[:,:,0].reshape(3*(ntr))
            
            self.parse_arr_to_sys(t_reinit, yini, temp)

            return t_reinit, yini, temp
        self.initial_conditions = staticmethod(MbM_initial_conditions)
        return
    
    ### Define ODE ###
    
    @staticmethod
    def update_vals(t : float, y : np.ndarray, vals : BGSystem):
        """
        Translate an array of data at time t into a `BGSystem`.

        Parameters
        ----------
        t : float
            time coordinate
        y : NDArray:
            array of data at time t
        vals : BGSystem
            the target system
        atol : float
            absolute tolerance parameters (if needed)
        rtol : float
            relative tolerance parameters (if needed)
        """

        #parse dynamical variables back to vals:
        vals.t.value = t
        vals.N.value = y[0]

        #compute static variables from y:
        vals.a.value = (np.exp(y[0]))

        #in case heaviside functions are needed in update_vals, atol and rtol are passed
        return
    
    @staticmethod
    def timestep(t : float, y : np.ndarray, vals : BGSystem) -> np.ndarray:
        """
        Compute time derivatives for data at time t using a `BGSystem`.

        Parameters
        ----------
        t : float
            a time coordinate
        y : NDArray:
            an array of data at time t
        vals : BGSystem
            the system used to compute the derivative
        atol : float
            absolute tolerance parameters (if needed)
        rtol : float
            relative tolerance parameters (if needed)

        Returns
        -------
        dydt : NDArray
            the time derivative of y
        """
        dydt = np.zeros_like(y)

        dydt[0] = vals.H.value #use the constant hubble rate to evolve N
        return dydt
    
    def ode(self, t : float, y : np.ndarray, vals : BGSystem) -> np.ndarray:
        """
        subsequently call `update_vals` and `timestep` to formulate an ODE for `solve_ivp`.

        Parameters
        ----------
        t : float
            a time coordinate
        y : NDArray:
            an array of data at time t
        vals : BGSystem
            the system passed to `update_vals` and `timestep` and 

        Returns
        -------
        dydt : NDArray
            the time derivative of y
        """
        self.update_vals(t, y, vals)
        dydt = self.timestep(t, y, vals)
        return dydt
    
    ### solve EoMs ###
    
    def solve_eom(self, t0 : float, yini : np.ndarray, vals : BGSystem):
        """
        Attempt to solve the GEF EoM's using `scipy.integrate.solve_ivp`.

        The solver attempts to obtain a GEF solution. This solution is then checked for any `ErrorEvent` occurrences.
        In this case, the solver marks the solution as unsuccessful and returns it for further processing.
        If no `ErrorEvent` occurrences are found, the other `TerminalEvent` occurrences are analyzed.
        These decide if the solution is returned, repeated, or if the solver should continue to solve.
        If there are no active `TerminalEvent` instances, the solution is returned after reaching `settings['tend']`.

        Parameters
        ----------
        t0 : float
            time of initialization
        yini : NDArray
            initial data at t0
        vals : BGSystem
            evolved alongside yini

        Returns
        -------
        sol
            a bunch object returned by `solve_ivp` containing the solution

        Raises
        ------
        TruncationError: 
            if an internal error occurred while solving the ODE.
        RuntimeError:
            if no 'finish' or 'error' command is obtained from an `Event` and `settings['tend']` is also not reached.
        """
        done = False
        attempts = 0
        sols = []

        event_dict, event_funcs = self._setup_events()

        print(f"The GEFSolver aims at reaching t={self.tend} with ntr={self.ntr}.")
        while not(done) and attempts < 10:

            try:
                tend = self.tend
                atol = self.settings["atol"]
                rtol = self.settings["rtol"]
                solvermethod = self.settings["solvermethod"]

                teval = np.arange(np.ceil(10*t0), np.floor(10*tend) +1)/10 #hotfix

                sol = solve_ivp(self.ode, [t0,tend], yini, t_eval=teval, args=(vals,),
                                method=solvermethod, atol=atol, rtol=rtol, events=event_funcs)
                if not(sol.success):
                    raise TruncationError
            #find 
            except KeyboardInterrupt:
                raise KeyboardInterrupt(f"Interrupted solver at t={float(vals.t.value):.2f}, N={float(vals.N.value):.2f}.")
            except Exception as e:
                print(f"While solving the GEF ODE, an error occurred at t={float(vals.t.value):.2f}, N={float(vals.N.value):.2f}):")
                print(e)
                raise TruncationError
            
            else:
                sols.append(sol)

                event_dict_new, command, terminal_event = self._assess_event_occurrences(sol.t_events, sol.y_events, vals)

                for key in event_dict_new.keys():
                    event_dict[key].append(event_dict_new[key])

                if command in ["finish", "error"]:
                    done=True
                elif command=="repeat":
                    print("Repeating")
                    sols.pop()
                elif command=="proceed":
                    #print("Proceeding")
                    t0 = sol.t[-1]
                    yini = sol.y[:,-1]

        sol = self._finalise_solution(sols, event_dict, terminal_event)
    
        if attempts != 1 and not(done):
            print(f"The run failed after {attempts} attempts.")
            raise RuntimeError
        
        return sol

    
    def _setup_events(self):
        event_funcs = []
        event_dict = {}
        for name, event in self.known_events.items():
            if event.active:
                event_funcs.append(event.event_func)
                event_dict[name] = []
        
        return event_dict, event_funcs
    
    
    def _assess_event_occurrences(self, t_events, y_events, vals):
        
        event_dict = {}

        active_events = [event for event in self.known_events.values() if event.active]

        # record which events have occurred
        occurrences = {}
        for i, event in enumerate(active_events):

            #Check if the event occured
            occurrence = (len(t_events[i]) != 0)

            #Add the event occurrence to the event dictionary:
            if occurrence:
                event_dict.update({event.name:t_events[i]})
                if len(t_events[i])>1:
                    tstr = [f"{t:.1f}" for t in t_events[i]]
                else:
                    tstr = f"{t_events[i][0]:.1f}"
                print(f"{event.name} at t={tstr}")
            occurrences[event.name] = occurrence

        error_events = [event for event in active_events if event.type=="error"]
        
        
        # Treat occurrences of ErrorEvents 
        for event in error_events:
            if occurrences[event.name]:
                print(f"Error: {event.message}")
                return event_dict, "error", event.name
                
        # if not ErrorEvents occurred, treat TerminalEvents
        terminal_events = [event for event in active_events if event.type=="terminal"]
        commands = {"primary":[], "secondary":[]}
        
        for event in terminal_events:
            #Asses the events consequences based on its occurrence or non-occurrence
            primary, secondary = event.event_consequence(vals, occurrences[event.name])
                
            for key, item in {"primary":(primary, event.name), "secondary":secondary}.items(): 
                commands[key].append(item)

        # If no error has occurred, handle secondary commands
        for command in commands["secondary"]:
            for key, item in command.items():
                if key =="timestep":
                    setattr(self, key, item)
                elif key=="tend":
                    if item > self.tend:
                        print(f"Increasing tend by {item-self.tend:.1f} to {item:.1f}")
                        setattr(self, key, item)
                elif key in self.settings.keys():
                    self.update_settings({key:item})
                else:
                    print(f"Unknown setting '{key}', ignoring input.")
        
        #Check command priority (in case of multiple terminal events). finish > repeat > proceed
        for primary_command in ["finish", "repeat", "proceed"]:
            for item in commands["primary"]:
                command = item[0]
                trigger = item[1]
                if command == primary_command:
                    return event_dict, command, trigger 

        #if no primarycommand was passed, return "finish" (no TerminalEvent or ErrorEvents)
        return event_dict, "finish", None
    
    def _finalise_solution(self, sols, event_dict, trigger_event):
        nfevs = 0
        y = []
        t = []
        solution = sols[-1]
        for s in sols:
            t.append(s.t[:-1])
            y.append(s.y[:,:-1])
            nfevs += s.nfev
        t.append(np.array([s.t[-1]]))
        y.append(np.array([s.y[:,-1]]).T)

        t = np.concatenate(t)
        y = np.concatenate(y, axis=1)

        solution.t = t
        solution.y = y
        solution.nfev = nfevs

        if trigger_event is None:
            solution.success = True
        else:
            solution.success = (trigger_event not in [event.name for event in self.known_events.values() if event.active and event.type=="error"])
            solution.message = f"A terminal event occured: '{trigger_event}'"

        for event_name in (event_dict.keys()):
            try:
                event_dict[event_name] = np.concatenate(event_dict[event_name])
            except ValueError:
                event_dict[event_name] = np.array(event_dict[event_name])

        solution.events = event_dict
        return solution
    
    ### Utility fumctions ###
    
    #can become an internal-only method once the GEFSolver output is changed.
    def parse_arr_to_sys(self, t : float, y : np.ndarray, vals : BGSystem):
        """
        Translate a GEF solution array to a BGSystem.

        Parameters
        ----------
        t : NDArray
            array of time coordinates
        y : NDArray
            array of variables at time t
        vals : BGSystem
            the target system
        """
        vals.units = False
        self.update_vals(t, y, vals)
        return
    
    def update_settings(self, **new_settings):
        """
        Update the `settings` of the class.
        """
        unknown_settings = []
        
        for setting, value in new_settings.items():
            if setting not in self.settings.keys():
                unknown_settings.append(setting)
            elif value != self.settings[setting]:
                print(f"Changing '{setting}' from {self.settings[setting]} to {value}.")
                self.settings[setting] = value
                if setting in ["atol", "rtol"]:
                    setattr(AuxTol, setting, value)
        
        if len(unknown_settings) > 0:
            print(f"Unknown settings: {unknown_settings}")
        return
    
    def _increase_ntr(self, val : int):
        self.ntr+=val
        print(f"Increasing 'ntr' by {val} to {self.ntr}.")
        return

    def _test_ode(self):
        try:
            t, yini, vals = self.initial_conditions()
        except Exception:
            raise ODECompilationError("Error while computing initial conditions.")
        try:
            self.ode(t, yini, vals)
        except Exception:
            raise ODECompilationError("Error while trying to compute a single ODE timestep.")
    
    
def GEFSolver(new_init : Callable, new_update_vals : Callable, new_timestep : Callable, new_variables : dict, new_events : list['Event'] = []):
    
    class CustomGEFSolver(BaseGEFSolver):
        vals_to_yini = staticmethod(new_init)
        update_vals = staticmethod(new_update_vals)
        timestep = staticmethod(new_timestep)
        known_variables = new_variables
        known_events = {event.name : event for event in new_events}
    
    CustomGEFSolver.__qualname__ = "GEFSolver"
    CustomGEFSolver.__module__ = __name__
    return CustomGEFSolver
        
    
class Event:
    def __init__(self, name : str, eventtype : str, func : Callable, terminal : bool, direction : int):
        """
        Initialise the event as `active`.

        Parameters
        ----------
        name : str
            sets the `name` attribute
        eventtype : str
            sets the `type` attribute
        func : Callable
            sets the `event_func` attribute
        terminal : boolean
            defines if the event occurrence is terminal or not
        direction : int
            defines the direction for which event occurrences are tracked
        """
        self.name : str= name
        """The name of the event."""

        self.type : str= eventtype
        """The event type 'terminal', 'error', or 'observer'."""
    
        self.active : bool = True
        """The events state, `False` implies the `Event` is disregarded by the solver."""

        func.terminal = terminal
        func.direction = direction
        
        self.event_func = func
        return
    
    @staticmethod
    def event_func(t : float, y : np.ndarray, sys : BGSystem) -> float:
        """
        The event tracked by `Event`

        This method is overwritten by the `func` input upon class initialisation.
        The signature and return of `func`needs to match this method

        Parameters
        ----------
        t : float
            time
        y : np.ndarray
            the solution array
        sys : BGSystem
            the system which is evolved alongside y

        Returns
        -------
        condition : float
            condition=0 is an event occurrence
        """
        return 1.
    
class TerminalEvent(Event):
    """
    An `Event` subclass whose occurrence terminates the solver.

    When the solver has terminated (due to an event or otherwise) it checks for `TerminalEvent` occurrences.
     This calls the `event_consequence` method, which returns instructions to `BaseGEFSolver`. 
     These instructions may be different depending on an event occurrence or a non-occurrence.
    """
    def __init__(self, name : str, func : Callable, direction : int, consequence : Callable):
        """
        Initialise the parent class and overwrite the `event_consequence` method.

        Parameters
        ----------
        name : str
            sets the `name` attribute
        func : Callable
            sets the `event_func` attribute
        direction : int
            passed as `direction` to the parent class
        consequence : Callable
            overwrites the `event_consequence` method
        """
        super().__init__(name, "terminal", func, True, direction)
        #ToDo: check for Consequence signature (once it is fixed)
        self.event_consequence = staticmethod(consequence)

    @staticmethod
    def event_consequence(sys : BGSystem, occurrence : bool) -> Tuple[str, dict]:
        """
        Inform the solver how to handle a (non-)occurrence of the event.
        
        This method is overwritten by the `consequence` input upon class initialisation.

        The methods returns are treated as an instruction to the solver:
        - `primary`: this informs the solver what to do with its ODE solution:
            - 'finish': the solver returns its solution marked as successful.
            - 'proceed': the solver continues solving from the termination time onwards.
            - 'repeat': the solver recomputes the solution.
        - `secondary`: this informs the solver if any of its settings need to be changed. 
        Allowed attributes are 'timestep', 'tend', 'atol', 'rtol'. See `BaseGEFSolver` for more details.

        Parameters
        ----------
        sys : BGSystem
            a system containing the solution of the solver
        occurrence : bool
            indicates if the event occurred during the solution or not

        Returns
        -------
        primary : str
            either 'finish', 'proceed' or 'repeat'
        secondary : dict
            the affected settings as keys and their new value as an item
        """
        if occurrence:
            return "proceed", {}
        else:
            return "proceed", {}


class ErrorEvent(Event):
    """
    An `Event` subclass whose occurrence indicates undesired behavior of the solution.

    When the solver terminates with an `ErrorEvent`, `BaseGEFSolver.solve_eom` returns the solution as unsuccessful.
    """
    def __init__(self, name : str, func : Callable, direction : int, message : str):
        """Initialise the parent class.
        
        The additional parameter `message` is printed on an event occurrence
        """
        super().__init__(name, "error", func, True, direction)
        self.message = message

class ObserverEvent(Event):
    """An `Event` which does not terminate the solve, but only records its occurences."""
    def __init__(self, name : str, func : Callable, direction : int):
        """Initialise the parent class."""
        super().__init__(name, "observer", func, False, direction)

class TruncationError(Exception):
    """
    Exception indicating that a GEF solution was unsuccessful.
    """
    pass

class ODECompilationError(Exception):
    """
    Exception indicating that an error occured while testing the ODE of a model.
    """
    pass

#define longer method docs from docs_solver:
generate_docs(docs_solver.DOCS)