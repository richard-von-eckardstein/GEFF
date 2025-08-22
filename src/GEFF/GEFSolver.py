import numpy as np

from GEFF.bgtypes import BGSystem

from scipy.integrate import solve_ivp

from copy import deepcopy

from typing import Callable

class TruncationError(Exception):
    pass

class Event:
    """
    An event which is tracked while solving the GEF equations.

    The class defines a function `event(t, y)` which is used by `scipy.integrate.solve_ivp` to track occurrences of `event(t, y(t))=0`.
    The event can be `terminal` causing the solver to stop upon `event(t, y(t))=0`.
    The event only triggers if the event condition changes sign according to:
    - positive zero crossing: `direction=1`
    - negative derivative, `direction=-1` 
    - arbitrary zero crossing `direction=0`

    The zeros are recorded and returned as part of the solvers output.

    Within the `GEFSolver` class, three subclasses of `Event` are used by the `solve_GEF` method:
    1. `TerminalEvent`
    2. `ErrorEvent`
    3. `ObserverEvent` 

    Attributes
    ----------
    `name` : str
        the name of the event
    `eventtype` : str
        the eventtype 'terminal', 'error', or 'observer'
    event_func : Callable
        the event condition
    active : boolean
        The events state. `False` implies the `Event` is disregarded by `GEFSolver`.
    """
    
    def __init__(self, name : str, eventtype : str, func : Callable, terminal : bool, direction : int):
        """
        Initialise the event as `active`.

        Parameters
        ----------
        name : str
            sets the `name` attribute
        eventtype : str
            sets the `eventtype` attribute
        func : Callable
            sets the `event_func` attribute
        terminal : boolean
            defines if the event occurrence is terminal or not
        direction : int
            defines the direction for which event occurrences are tracked.
        """
        self.name = name

        self.type = eventtype
        
        func.terminal = terminal
        func.direction = direction
        
        self.event_func = staticmethod(func)
    
        self.active = True
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
     This calls the `event_consequence` method, which returns instructions to the `GEFSolver`. 
     These instructions may be different depending on an event occurrence or a non-occurrence.
    """
    def __init__(self, name : str, func : Callable, direction : int, consequence : Callable):
        """
        Initialise the parent class and overwrite the `event_consequence` method.

        Parameters
        ----------
        name : str
            passed as `name` to the parent class
        func : Callable
            passed as `func` to the parent class
        direction : int
            passed as `direction` to the parent class
        consequence : Callable
            overwrites the `event_consequence` method
        """
        super().__init__(name, "terminal", func, True, direction)
        #ToDo: check for Consequence signature (once it is fixed)
        self.event_consequence = staticmethod(consequence)

    @staticmethod
    def event_consequence(sys : BGSystem, occurrence : bool) -> tuple[str, dict]:
        """
        Inform the solver how to handle a (non-)occurrence of the event.
        
        This method is overwritten by the `consequence` input upon class initialisation.

        The methods returns are treated as an instruction to the solver:
        - `primary`: this informs the solver what to do with its ODE solution:
            - 'finish': the solver returns its solution marked as successful.
            - 'proceed': the solver continues solving from the termination time onwards.
            - 'repeat': the solver recomputes the solution.
        - `secondary`: this informs the solver if any of its settings need to be changed. 
        Allowed settings are 'timestep', 'tend', 'atol', 'rtol'. See `GEFSolver` for more details.

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
            the affected settings as keys and their new value as an item.
        """
        if occurrence:
            return "proceed", {}
        else:
            return "proceed", {}


class ErrorEvent(Event):
    """
    An `Event` subclass whose occurrence indicates undesired behavior of the solution.

    When the solver terminates with an `ErrorEvent`, the `GEFSolver` returns the solution as unsuccessful.
    """
    def __init__(self, name, func, direction):
        """Initialise the parent class."""
        super().__init__(name, "error", func, True, direction)

class ObserverEvent(Event):
    """An `Event` which does not terminate the solver and is only recorded."""
    def __init__(self, name, func, direction):
        """Initialise the parent class."""
        super().__init__(name, "observer", func, False, direction)


class GEFSolver:
    """
    A class used to solve the GEF equations defined by a GEF model.

    Attributes
    ----------
    
    """
    def __init__(self, update_vals, timestep, initialise, events, variable_dict):
        self._base_initialise = initialise
        #think how you can add these as (static)methods to the class -> helps documentation
        self.compute_initial_conditions = self.initialise_from_slowroll
        self._update_vals = update_vals
        self.compute_timestep = timestep

        self.solver_events = {event.name: event for event in events}

        #self.variable_classification = variable_dict

        self.settings={"atol":1e-20, "rtol":1e-6, "GEFattempts":5, "GEFmethod":"RK45", "ntrstep":10}
        self.ntr = 100
        self.tend = 120

    def set_init_vals(self, initsys):
        self.init_vals = BGSystem.from_system(initsys, copy=True)
        return
    
    def update_settings(self, **new_settings):
        unknown_settings = []
        
        for setting, value in new_settings.items():
            if setting not in self.settings.keys():
                unknown_settings.append(setting)
            elif value != self.settings[setting]:
                print(f"Changing {setting} from {self.settings[setting]} to {value}.")
                self.settings[setting] = value
        
        if len(unknown_settings) > 0:
            print(f"Unknown settings: {unknown_settings}")
        return
    
    def _increase_ntr(self, val):
        self.ntr+=val
        print(f"Increasing ntr by {val} to {self.ntr}.")
        return
        
    #stays part of the solver
    def _ode(self, t, y, vals):
        atol = self.settings["atol"]
        rtol = self.settings["rtol"]
        self._update_vals(t, y, vals, atol=atol, rtol=rtol)
        dydt = self.compute_timestep(t, y, vals, atol=atol, rtol=rtol)
        return dydt
    
    #Can stay in the Solver
    def compute_GEF_solution(self):
        #initial configuration
        maxntr = 200
        maxattempts = self.settings["GEFattempts"]
        ntrstep = self.settings["ntrstep"]

        attempt=0
        done = False
        sol = None
        #Run GEF
        while not(done) and (attempt<maxattempts):
            attempt+=1
            try:
                #Try to get convergent solution
                t0, yini, vals = self.compute_initial_conditions()
                sol = self.solve_GEF(t0, yini, vals)
                done = sol.success
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except TruncationError:
                done = False
            
            #Standard response to error: increase ntr (maximum: 200)
            if not(done):
                if self.ntr<maxntr:
                    self._increase_ntr( min(ntrstep, maxntr-self.ntr) )
                else:
                    print("Cannot increase ntr further.")
                    break
        
        if not(done):
            print(f"The run did not converge after {attempt} attempts.")

            if sol is None:
                raise RuntimeError(f"Not a single successful solution after {attempt} attempts.")
            else:
                sol
                print("Processing last solution.\n")
                return sol, vals
                
        else: 
            print("Run converged.\n")

        return sol, vals
    
    #move to GEF
    def MbMcrosscheck(self, spec, vals, errthr, binning, method, **MbMkwargs):
        errs, terr, _ = spec.estimate_GEF_error(vals, errthr=errthr, binning=binning, method=method, **MbMkwargs)

        reinit_inds = []
        agreement=True
        for err in errs:
            rmserr = np.sqrt(np.sum(err**2)/len(err))
            if max(err[-1], rmserr) > 0.10:
                agreement=False
                #find where the error is above 5%, take the earliest occurrence, reduce by 1
                inds = np.where(err > errthr)
                err_ind = inds[0][0]-1               
            else:
                err_ind = len(terr)-1
            reinit_inds.append( err_ind )

        t0 = terr[min(reinit_inds)]

        ind = np.where(spec["t"] <= t0)[0][-1]

        reinit_slice = spec.tslice(ind)

        return agreement, reinit_slice
    
    #can be moved to Solution?
    def update_sol(self, solold, solnew):
        if solold is None:
            return solnew
        else:
            sol = solold
            ind_overlap = np.where(solnew.t[0] >= solold.t)[0][-1]
            sol.t = np.concatenate([solold.t[:ind_overlap], solnew.t])

            if solold.y.shape[0] < solnew.y.shape[0]:
                #if ntr increased from one solution to the next, fill up solold with zeros to match solnew
                fillshape = (solnew.y.shape[0] - solold.y.shape[0], solold.y.shape[1])
                yfill = np.zeros( fillshape )
                solold.y = np.concatenate([solold.y, yfill], axis=0)

            sol.y = np.concatenate([solold.y[:,:ind_overlap], solnew.y], axis=1)
            sol.events.update(solnew.events)
            for attr in ["nfev", "njev", "nlu"]:
                setattr(sol, attr, getattr(solold, attr) + getattr(solnew, attr))
            for attr in ["message", "success", "status"]:
                setattr(sol, attr, getattr(solnew, attr))
            return sol
    
    #stays in Solver
    def initialise_from_slowroll(self):
        t0 = 0
        vals = deepcopy(self.init_vals)
        vals.set_units(False)
        yini = self._base_initialise(vals, self.ntr)
        return t0, yini, vals
    
    #stays in solver
    def initialise_from_MbM(self, sol, reinit_spec, method, **MbMkwargs):
        def new_initialiser():
            ntr = self.ntr

            t_reinit = reinit_spec["t"]

            reinitInd = np.where(sol.t == t_reinit)[0]

            #Create unit system:
            temp = BGSystem.from_system(self.init_vals, copy=True)

            #Construct yini from interpolation:
            ytmp = sol.y[:,reinitInd]

            #Parse yini to temp
            self.parse_arr_to_sys(t_reinit, ytmp, temp)

            #Use "Initialise" to zero out all GEF-bilinear values
            yini = self._base_initialise(temp, ntr)
            gaugeinds = np.where(yini==0.)[0]

            #parse back E0, B0, G0 (assuming they are at the same spot, should be the case.)
            for i in range(len(yini)):
                if i not in gaugeinds[3:]:
                    yini[i] = ytmp[i]

            # compute En, Bn, Gn, for n>1 from Modes
            yini[gaugeinds[3:]] = np.array(
                                    [
                                    reinit_spec.IntegrateSpecSlice(n=n, method=method,**MbMkwargs)
                                    for n in range(1,ntr+1)
                                    ]
                                    )[:,:,0].reshape(3*(ntr))
            
            self.parse_arr_to_sys(t_reinit, yini, temp)

            return t_reinit, yini, temp
        return new_initialiser
        
    #Taken care of by Map
    def parse_arr_to_sys(self, t, y, vals):
        ts = deepcopy(t)
        ys = deepcopy(y)
        vals.set_units(False)
        self._update_vals(ts, ys, vals)
        return
    
    def solve_GEF(self, t0, yini, vals):
        done = False
        attempts = 0
        sols = []

        event_dict, event_funcs = self._setup_events()

        print(f"The solver aims at reaching t={self.tend} with ntr={self.ntr}.")
        while not(done) and attempts < 10:

            try:
                tend = self.tend
                atol = self.settings["atol"]
                rtol = self.settings["rtol"]
                solvermethod = self.settings["GEFmethod"]

                teval = np.arange(np.ceil(10*t0), np.floor(10*tend) +1)/10 #hotfix

                sol = solve_ivp(self._ode, [t0,tend], yini, t_eval=teval, args=(vals,),
                                method=solvermethod, atol=atol, rtol=rtol, events=event_funcs)
                if not(sol.success):
                    raise ValueError
            except KeyboardInterrupt:
                print(f"The run failed at t={vals.t}, N={vals.N}.")
                raise KeyboardInterrupt
            except Exception as e:
                print(f"While solving the GEF ODE, an error occured at t={vals.t}, N={vals.N} : \n \t {e}")
                raise TruncationError
            
            else:
                sols.append(sol)

                event_dict_new, command, terminal_event = self._assess_event_occurrences(sol.t_events, sol.y_events, vals)

                for key in event_dict_new.keys():
                    event_dict[key]["t"].append(event_dict_new[key]["t"])
                    event_dict[key]["N"].append(event_dict_new[key]["N"])

                if command in ["finish", "error"]:
                    done=True
                elif command=="repeat":
                    print("Repeating")
                    sols.pop()
                elif command=="proceed":
                    #print("Proceeding")
                    t0 = sol.t[-1]
                    yini = sol.y[:,-1]

        self._finalise_solution(sols, event_dict, terminal_event)
    
        if attempts != 1 and not(done):
            print(f"The run failed after {attempts} attempts.")
            raise RuntimeError
        
        return sol
    
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
            solution.success = (trigger_event not in [event.name for event in self.solver_events.values() if event.active and event.type=="error"])
            solution.message = f"A terminal event occured: '{trigger_event}'"

        for event_name in (event_dict.keys()):
            try:
                event_dict[event_name]["t"] = np.round(np.concatenate(event_dict[event_name]["t"]), 1)
                event_dict[event_name]["N"] = np.round(np.concatenate(event_dict[event_name]["N"]), 3)
            except ValueError:
                event_dict[event_name]["t"] = np.array(event_dict[event_name]["t"])
                event_dict[event_name]["N"] = np.array(event_dict[event_name]["N"])

        solution.events = event_dict
        return solution
    
    ### Handling of events ###
    ##########################

    def toggle_event(self, event_name, toggle):
        if event_name in [event for event in self.solver_events.keys()]:
            self.solver_events[event_name].active = toggle
            humanreadable = {True:"active", False:"inactive"}
            print(f"The event '{event_name}' is now {humanreadable[toggle]}")
        else:
            print(f"Unknown event: '{event_name}'")
        return

    def _setup_events(self):
        
        #eventually reinstate this
        """
        def EventWrapper(t, y, vals):
            def SolveIVPcompatibleEvent(func):
                return func(t, y, vals, self.settings["atol"], self.settings["rtol"])
            return SolveIVPcompatibleEvent
        """

        event_funcs = []
        event_dict = {}
        for name, event in self.solver_events.items():
            if event.active:
                event_funcs.append(event.event_func)
                event_dict[name] = {"t":[], "N":[]}
        
        return event_dict, event_funcs
    
    def _assess_event_occurrences(self, t_events, y_events, vals):
        commands = {"primary":[], "secondary":[]}
        event_dict = {}

        active_events = [event for event in self.solver_events.values() if event.active]

        for i, event in enumerate(active_events):

            #Check if the event occured
            occurrence = (len(t_events[i]) != 0)

            #Add the event occurrence to the event dictionary:
            if occurrence:
                event_dict.update({event.name:{"t":t_events[i], "N": y_events[i][:,0]}})
                print(f"{event.name} at t={np.round(t_events[i], 1)} and N={np.round(y_events[i][:,0],1)}.")

            if event.type == "error" and occurrence:
                commands["primary"].append( ("error", event.name) )
            
            elif event.type=="terminal":
                #Asses the events consequences based on its occurrence or non-occurrence
                primary, secondary = event.event_consequence(vals, occurrence)
                
                for key, item in {"primary":(primary, event.name), "secondary":secondary}.items(): 
                    commands[key].append(item)


        #Handle secondary commands
        for command in commands["secondary"]:
            for key, item in command.items():
                if key in ["timestep", "tend", "atol", "rtol"]:
                    setattr(self, key, item)
                else:
                    print("Unknown setting 'key', ignoring input.")
        
        #Check command priority (in case of multiple final events occuring). error > finish > repeat > proceed
        for primary_command in ["error", "finish", "repeat", "proceed"]:
            for item in commands["primary"]:
                command = item[0]
                trigger = item[1]
                if command == primary_command:
                    return event_dict, command, trigger 

        #if no primarycommand was passed, return "finish"
        return event_dict, "finish", None
    
    
        
    