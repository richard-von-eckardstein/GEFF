import numpy as np

from GEFF.BGTypes import BGSystem

from scipy.integrate import solve_ivp

from copy import deepcopy

class TruncationError(Exception):
    pass

class EventError(Exception):
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
        self.EventFunc = func
    
        self.active = True
        return
    
class TerminalEvent(Event):
    def __init__(self, name, func, direction, consequence):
        super().__init__(name, "terminal", func, True, direction)
        #ToDo: check for Consequence signature (once it is fixed)
        self.EventConsequence = consequence

class ErrorEvent(Event):
    def __init__(self, name, func, direction):
        super().__init__(name, "error", func, True, direction)

class ObserverEvent(Event):
    def __init__(self, name, func, direction):
        super().__init__(name, "observer", func, False, direction)




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


class GEFSolver:
    def __init__(self, UpdateVals, TimeStep, Initialise, events, VariableDict):
        self.__Initialise = Initialise
        self.InitialConditions = self.InitialiseFromSlowRoll
        
        self.__UpdateVals = UpdateVals
        self.TimeStep = TimeStep

        self.Events = events

        self.VariableClassification = VariableDict

        self.settings={"atol":1e-20, "rtol":1e-6, "reachNend":True, "GEFattempts":5, "solmeth":"RK45"}
        self.ntr = 100
        self.tend = 120

    def SetIniVals(self, initsys):
        self.iniVals = BGSystem.FromSystem(initsys, copy=True)
        return
    
    def UpdateSettings(self, **newsettings):
        unknownsettings = []
        
        for setting, value in newsettings.items():
            if setting not in self.settings.keys():
                unknownsettings.append(setting)
            elif value != self.settings[setting]:
                print(f"Changing {setting} from {self.settings[setting]} to {value}.")
                self.settings[setting] = value
        
        if len(unknownsettings) > 0:
            print(f"Unknown settings: {unknownsettings}")
        return
    
    def IncreaseNtr(self, val=10):
        self.ntr+=val
        print(f"Increasing ntr by {val} to {self.ntr}.")
        return
        
    #stays part of the solver
    def __ode(self, t, y, vals):
        atol = self.settings["atol"]
        rtol = self.settings["rtol"]
        self.__UpdateVals(t, y, vals, atol=atol, rtol=rtol)
        dydt = self.TimeStep(t, y, vals, atol=atol, rtol=rtol)
        return dydt
    
    #Can stay in the Solver
    def GEFAlgorithm(self, ntrstep=10):
        maxntr = 200
        maxattempts = self.settings["GEFattempts"]
        attempt=0
        done = False
        #Run GEF
        while not(done) and (attempt<maxattempts):
            attempt+=1
            try:
                t0, yini, vals = self.InitialConditions()
                sol, done = self.SolveGEF(t0, yini, vals)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except TruncationError:
                done = False
            
            if not(done):
                if self.ntr<=maxntr:
                    self.IncreaseNtr( min(ntrstep, maxntr-self.ntr) )
                else:
                    print("Cannot increase ntr further.")
                    break
        
        if not(done):
            print(f"The run did not converge after {attempt} attempts.")
            try:
                sol
                print("Proceeding with last solution.")
                return sol, vals
            except:
                raise RuntimeError(f"Not a single successful solution after {attempt} attempts.")

        return sol, vals
    
    #move to GEF
    def ModeByModeCrossCheck(self, spec, vals, errthr, thinning, method, **MbMKwargs):
        errs, terr, _ = spec.CompareToBackgroundSolution(vals, errthr=errthr, steps=thinning, method=method, **MbMKwargs)

        reinitinds = []
        agreement=True
        for err in errs:
            rmserr = np.sqrt(np.sum(err**2)/len(err))
            if max(err[-1], rmserr) > 0.10:
                agreement=False
                #find where the error is above 5%, take the earliest occurance, reduce by 1
                inds = np.where(err > errthr)
                errInd = inds[0][0]-1               
            else:
                errInd = len(terr)-1
            reinitinds.append( errInd )

        t0 = terr[min(reinitinds)]

        ind = np.where(spec["t"] <= t0)[0][-1]

        ReInitSlice = spec.TSlice(ind)

        return agreement, ReInitSlice
    
    #can be moved to Solution?
    def UpdateSol(self, solold, solnew):
        if solold==None:
            return solnew
        else:
            sol = solold
            indoverlap = np.where(solnew.t[0] >= solold.t)[0][-1]
            sol.t = np.concatenate([solold.t[:indoverlap], solnew.t])

            if solold.y.shape[0] < solnew.y.shape[0]:
                #if ntr increased from one solution to the next, fill up solold with zeros to match solnew
                fillshape = (solnew.y.shape[0] - solold.y.shape[0], solold.y.shape[1])
                yfill = np.zeros( fillshape )
                solold.y = np.concatenate([solold.y, yfill], axis=0)

            sol.y = np.concatenate([solold.y[:,:indoverlap], solnew.y], axis=1)
            sol.events.update(solnew.events)
            sol.nfev +=sol.nfev
            return sol
    
    #stays in Solver
    def InitialiseFromSlowRoll(self):
        t0 = 0
        vals = deepcopy(self.iniVals)
        vals.SetUnits(False)
        yini = self.__Initialise(vals, self.ntr)
        return t0, yini, vals
    
    #stays in solver
    def InitialiseFromMbM(self, sol, ReInitSpec, method, **MbMKwargs):
        def NewInitialiser():
            ntr = self.ntr

            treinit = ReInitSpec["t"]

            reinitInd = np.where(sol.t == treinit)[0]

            #Create unit system:
            Temp = deepcopy(self.iniVals)

            #Construct yini from interpolation:
            ytmp = sol.y[:,reinitInd]

            #Parse yini to Temp
            self.ParseArrToUnitSystem(treinit, ytmp, Temp)

            #Use "Initialise" to zero out all GEF-bilinear values
            yini = self.__Initialise(Temp, ntr)
            gaugeinds = np.where(yini==0.)[0]

            #parse back E0, B0, G0 (assuming they are at the same spot, should be the case.)
            for i in range(len(yini)):
                if i not in gaugeinds[3:]:
                    yini[i] = ytmp[i]

            # compute En, Bn, Gn, for n>1 from Modes
            yini[gaugeinds[3:]] = np.array(
                                    [
                                    ReInitSpec.IntegrateSpecSlice(n=n, method=method,**MbMKwargs)
                                    for n in range(1,ntr+1)
                                    ]
                                    )[:,:,0].reshape(3*(ntr))
            
            self.ParseArrToUnitSystem(treinit, yini, Temp)

            return treinit, yini, Temp
        return NewInitialiser
        
    #Taken care of by Map
    def ParseArrToUnitSystem(self, t, y, vals):
        ts = deepcopy(t)
        ys = deepcopy(y)
        vals.SetUnits(False)
        self.__UpdateVals(ts, ys, vals)
        return
    
    def SolveGEF(self, t0, yini, vals):
        done = False
        attempts = 0
        sols = []

        eventdic, eventfuncs = self.EventSetup()

        print(f"The solver aims at reaching t={self.tend} with ntr={self.ntr}.")
        while not(done) and attempts < 10:

            try:
                tend = self.tend
                atol = self.settings["atol"]
                rtol = self.settings["rtol"]
                solvermethod = self.settings["solmeth"]

                teval = np.arange(np.ceil(10*t0), np.floor(10*tend) +1)/10 #hotfix

                sol = solve_ivp(self.__ode, [t0,tend], yini, t_eval=teval, args=(vals,),
                                method=solvermethod, atol=atol, rtol=rtol, events=eventfuncs)
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

                eventdic_new, command = self.__AssessEvents(sol.t_events, sol.y_events, vals)

                for key in eventdic_new.keys():
                    eventdic[key]["t"].append(eventdic_new[key]["t"])
                    eventdic[key]["N"].append(eventdic_new[key]["N"])

                if command=="finish":
                    status=True
                    done=True
                elif command=="error":
                    status=False
                    done=True
                elif command=="repeat":
                    print("Repeating")
                    sols.pop()
                elif command=="proceed":
                    #print("Proceeding")
                    t0 = sol.t[-1]
                    yini = sol.y[:,-1]

        self.__FinaliseSolution(sols, eventdic)
    
        if attempts != 1 and not(done):
            print(f"The run failed after {attempts} attempts.")
            raise RuntimeError
        
        return sol, status
    
    def __FinaliseSolution(self, sols, eventdic):
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

        for eventname in (eventdic.keys()):
            try:
                eventdic[eventname]["t"] = np.round(np.concatenate(eventdic[eventname]["t"]), 1)
                eventdic[eventname]["N"] = np.round(np.concatenate(eventdic[eventname]["N"]), 3)
            except ValueError:
                eventdic[eventname]["t"] = np.array(eventdic[eventname]["t"])
                eventdic[eventname]["N"] = np.array(eventdic[eventname]["N"])

        solution.events = eventdic
        return solution
    
    ### Handling of events ###
    ##########################

    #ToDo's:
    #   - get rid of "reachNend" -> replace by Eventtoggle
    #   - implement handling of 'error' events


    def EventSetup(self):
        
        #eventually reinstate this
        def EventWrapper(t, y, vals):
            def SolveIVPcompatibleEvent(func):
                return func(t, y, vals, self.settings["atol"], self.settings["rtol"])
            return SolveIVPcompatibleEvent

        eventfuncs = []
        eventdic = {}
        for event in self.Events:
            if event.active:
                eventfuncs.append(event.func)
                eventdic[event.namee] = {"t":[], "N":[]}
        
        return eventdic, eventfuncs
    
    def __AssessEvents(self, tevents, yevents, vals):
        commands = {"primary":[], "secondary":[]}
        eventdic = {}

        activeEvents = [event for event in self.Events if event.active]

        for i, event in enumerate(activeEvents):

            #Check if the event occured
            occurance = (len(tevents[i]) != 0)

            #Add the event occurance to the event dictionary:
            if occurance:
                eventdic.update({event.name:{"t":tevents[i], "N": yevents[i][:,0]}})
                print(f"{event.name} at t={np.round(tevents[i], 1)} and N={np.round(yevents[i][:,0],1)}.")

            if event.type == "error" and occurance:
                commands["primary"].append("error")
            
            elif event.type=="terminal":
                #Asses the events consequences based on its occurance or non-occurance
                primary, secondary = event.EventConsequence(vals, occurance)
                
                for key, item in {"primary":primary, "secondary":secondary}.items(): 
                    commands[key].append(item)


        #Handle secondary commands
        for command in commands["secondary"]:
            for key, item in command.items():
                if key in ["TimeStep", "tend", "atol", "rtol"]:
                    setattr(self, key, item)
                else:
                    print("Unknown setting 'key', ignoring input.")
        
        #Check command priority (in case of multiple final events occuring). error > finish > repeat > proceed
        for primarycommand in ["error", "finish", "repeat", "proceed"]:
            if primarycommand in commands["primary"]:
                return eventdic, primarycommand

        #if no primarycommand was passed, return "finish"
        return eventdic, "finish"
    
    
        
    