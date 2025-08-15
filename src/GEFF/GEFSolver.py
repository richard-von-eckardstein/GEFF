import numpy as np

from GEFF.BGTypes import BGSystem

from scipy.integrate import solve_ivp

from copy import deepcopy

class TruncationError(Exception):
    pass

class Event:
    #Terminal events can modify a solver and pass concrete orders to the solver. The orders are:
    ### finish: finalise the solver
    ### repeat: repeat the last iteration of the solver
    ### proceed: continue solving from the point of termination
    #Any event may pass the secondary order "update" to the solver. This can update the following attributes of the solver:
    ### __ODE__, __tend, __atol, __rtol, __method 
    
    #Terminal event consequences should distinguish the cases: termination on this event, termination not on this event, termination without event.
    def __init__(self, name, eventtype, func, terminal, direction):
        self.name = name

        self.type = eventtype
        
        func.terminal = terminal
        func.direction = direction
        
        #ToDo: check for func signature (once it is fixed)
        self.event_func = func
    
        self.active = True
        return
    
class TerminalEvent(Event):
    def __init__(self, name, func, direction, consequence):
        super().__init__(name, "terminal", func, True, direction)
        #ToDo: check for Consequence signature (once it is fixed)
        self.event_consequence = consequence

class ErrorEvent(Event):
    def __init__(self, name, func, direction):
        super().__init__(name, "error", func, True, direction)

class ObserverEvent(Event):
    def __init__(self, name, func, direction):
        super().__init__(name, "observer", func, False, direction)


class GEFSolver:
    def __init__(self, update_vals, timestep, initialise, events, variable_dict):
        self._base_initialise = initialise
        self.compute_initial_conditions = self.initialise_from_slowroll
        
        self._update_vals = update_vals
        self.compute_timestep = timestep

        self.solver_events = {event.name: event for event in events}

        #self.variable_classification = variable_dict

        self.settings={"atol":1e-20, "rtol":1e-6, "GEFattempts":5, "GEFmethod":"RK45", "ntrstep":10}
        self.ntr = 100
        self.tend = 120

    def set_init_vals(self, initsys):
        self.init_vals = BGSystem.FromSystem(initsys, copy=True)
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
    def MbMcrosscheck(self, spec, vals, errthr, thinning, method, **MbMkwargs):
        errs, terr, _ = spec.CompareToBackgroundSolution(vals, errthr=errthr, steps=thinning, method=method, **MbMkwargs)

        reinit_inds = []
        agreement=True
        for err in errs:
            rmserr = np.sqrt(np.sum(err**2)/len(err))
            if max(err[-1], rmserr) > 0.10:
                agreement=False
                #find where the error is above 5%, take the earliest occurance, reduce by 1
                inds = np.where(err > errthr)
                err_ind = inds[0][0]-1               
            else:
                err_ind = len(terr)-1
            reinit_inds.append( err_ind )

        t0 = terr[min(reinit_inds)]

        ind = np.where(spec["t"] <= t0)[0][-1]

        reinit_slice = spec.TSlice(ind)

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
        vals.SetUnits(False)
        yini = self._base_initialise(vals, self.ntr)
        return t0, yini, vals
    
    #stays in solver
    def initialise_from_MbM(self, sol, reinit_spec, method, **MbMkwargs):
        def new_initialiser():
            ntr = self.ntr

            t_reinit = reinit_spec["t"]

            reinitInd = np.where(sol.t == t_reinit)[0]

            #Create unit system:
            temp = deepcopy(self.init_vals)

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
        vals.SetUnits(False)
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

                event_dict_new, command, terminal_event = self._assess_event_occurances(sol.t_events, sol.y_events, vals)

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
    
    def _assess_event_occurances(self, t_events, y_events, vals):
        commands = {"primary":[], "secondary":[]}
        event_dict = {}

        active_events = [event for event in self.solver_events.values() if event.active]

        for i, event in enumerate(active_events):

            #Check if the event occured
            occurance = (len(t_events[i]) != 0)

            #Add the event occurance to the event dictionary:
            if occurance:
                event_dict.update({event.name:{"t":t_events[i], "N": y_events[i][:,0]}})
                print(f"{event.name} at t={np.round(t_events[i], 1)} and N={np.round(y_events[i][:,0],1)}.")

            if event.type == "error" and occurance:
                commands["primary"].append( ("error", event.name) )
            
            elif event.type=="terminal":
                #Asses the events consequences based on its occurance or non-occurance
                primary, secondary = event.event_consequence(vals, occurance)
                
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
    
    
        
    