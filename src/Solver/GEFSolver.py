import numpy as np
from src.BGQuantities.BGTypes import BGSystem
from src.Models.Classic import Initialise as ClassicInitialiser
from src.Models.Classic import UpdateVals as ClassicComputStatic
from src.Models.Classic import TimeStep as ClassicEoM
from src.Models.Classic import events as ClassicEvents

from scipy.integrate import solve_ivp

from copy import deepcopy

class TruncationError(Exception):
    pass

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
    def __init__(self, UpdateVals, TimeStep, Initialise, events, ModeByMode):
        self.__Initialise = Initialise
        self.InitialConditions = self.InitialiseFromSlowRoll
        
        self.__UpdateVals = UpdateVals
        self.TimeStep = TimeStep

        self.Events = events

        self.ModeByMode = ModeByMode

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
        
    
    #stays part of the solver
    def __ode(self, t, y, vals):
        atol = self.settings["atol"]
        rtol = self.settings["rtol"]
        self.__UpdateVals(t, y, vals, atol=atol, rtol=rtol)
        dydt = self.TimeStep(t, y, vals, atol=atol, rtol=rtol)
        return dydt

    #is moved to GEF
    def RunGEF(self, ntr, tend, nmodes=500,
                 printstats=True, **Kwargs):
        self.ntr=ntr
        self.tend=tend

        GEFKwargs = {setting : Kwargs[setting] for setting in self.settings if setting in Kwargs}
        
        self.UpdateSettings(**GEFKwargs)

        MbMattempts = Kwargs.get("MbMattempts", 5)
        thinning = Kwargs.get("thinning", 5)
        errthr = Kwargs.get("errthr", 0.025)
        resumeMode = Kwargs.get("resumeMode", True)
        method = Kwargs.get("method", "simpson")
        selfcorrmethod = Kwargs.get("selfcorrmethod", "simpson")

        MbMKwargs = {"epsabs":self.settings["atol"], "epsrel":self.settings["rtol"]}

        done=False
        sol = None
        attempt=0
        while not(done) and attempt<MbMattempts:
            attempt +=1
            solnew, vals = self.GEFAlgorithm()
            sol = self.UpdateSol(sol, solnew)
            self.ParseArrToUnitSystem(sol.t, sol.y, vals)

            if nmodes!=None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeByMode(vals)

                if resumeMode:    
                    try:
                        spec["t"]
                    except:
                        spec = MbM.ComputeModeSpectrum(nmodes, rtol=self.settings["rtol"])
                    else:
                        spec = MbM.UpdateSpectrum(spec, treinit, rtol=self.settings["rtol"])
                else:
                    spec = MbM.ComputeModeSpectrum(nmodes, rtol=self.settings["rtol"])

                print("Performing mode-by-mode comparison with GEF results.")
                try:
                    treinit = ReInitSpec["t"]
                except:
                    treinit = 0
                agreement, ReInitSpec = self.ModeByModeCrossCheck(spec, vals, errthr=errthr, thinning=thinning, method=selfcorrmethod, **MbMKwargs)

                if agreement:
                    print(f"The mode-by-mode comparison indicates a convergent GEF run.")
                    done=True
                else:
                    Nreinit = np.round(ReInitSpec["N"], 1)
                    treinit = np.round(ReInitSpec["t"], 1)

                    print(f"Attempting to solve GEF using self-correction starting from t={treinit}, N={Nreinit}.")

                    self.InitialConditions = self.InitialiseFromMbM(sol, ReInitSpec, method, **MbMKwargs)
            else:
                spec=None
                done=True
        
        if done:
            print("GEF run successfully completed.")
            if printstats:
                PrintSummary(sol)
            return sol, spec
        else:
            raise RuntimeError(f"GEF did not complete after {attempt} attempts.")
    
    #Can stay in the Solver
    def GEFAlgorithm(self):
        maxattempts = self.settings["GEFattempts"]
        attempts=1
        done = False
        #Run GEF
        while not(done):
            try:
                t0, yini, vals = self.InitialConditions()
                sol = self.SolveGEF(t0, yini, vals)
            except TruncationError:
                attempts+=1
                if attempts > maxattempts:
                    break
                print("A truncation error occured")
                self.IncreaseNtr(10)
            else:
                done=True
        
        if attempts>maxattempts:
            print(f"The run did not finish after {maxattempts} attempts.")
            try:
                sol
                print("Proceeding with last successful solution.")
                return sol, vals
            except:
                raise RuntimeError(f"Not a single successful solution after {maxattempts} attempts.")

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
    
    #Stays in Solver
    def UnpackEvents(self):
        def EventWrapper(eventfunc):
            def SolveIVPcompatibleEvent(t, y, vals):
                return eventfunc(t, y, vals, self.settings["atol"], self.settings["rtol"])
            return SolveIVPcompatibleEvent

        eventfuncs = []
        eventdic = {}
        for event in self.Events:
            eventname = event.name
            if eventname == "End of inflation" and not(self.settings["reachNend"]):
                print("Removing default event 'End of inflation'")
            else:
                eventfuncs.append(event.func)
                eventdic[eventname] = {"t":[], "N":[]}
        return eventdic, eventfuncs
    
    def SolveGEF(self, t0, yini, vals):
        done = False
        attempts = 0
        sols = []

        eventdic, eventfuncs = self.UnpackEvents()

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
                print(f"Error MSG: {e}")
                print(f"The run failed at t={vals.t}, N={vals.N}.")
                raise TruncationError
            else:
                sols.append(sol)

                eventdic_new, command = self.__AssessEvents(sol.t_events, sol.y_events, vals)

                for key in eventdic_new.keys():
                    eventdic[key]["t"].append(eventdic_new[key]["t"])
                    eventdic[key]["N"].append(eventdic_new[key]["N"])

                if command=="finish":
                    print("Finishing")
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
        
        return sol
    
    #Stays in Solver
    def __AssessEvents(self, tevents, yevents, vals):
        commands = {"primary":[], "secondary":[]}
        eventdic = {}
        Events = deepcopy(self.Events)
        if not(self.settings["reachNend"]):
            #temporarily remove "End of inflation" from events
            for i, event in enumerate(Events):
                if event.name=="End of inflation":
                    Events.pop(i)

        for i, event in enumerate(Events):

            #Check if the event occured
            occurance = (len(tevents[i]) != 0)
            #Asses the events consequences based on its occurance or non-occurance
            consequence = event.EventConsequence(vals, occurance)
            for key, item in consequence.items(): 
                commands[key].append(item)
            #Add the event occurances to the event dictionary:
            if occurance:
                eventdic.update({event.name:{"t":tevents[i], "N": yevents[i][:,0]}})
                print(f"{event.name} at t={np.round(tevents[i], 1)} and N={np.round(yevents[i][:,0],1)}.")

        for command in commands["secondary"]:
            for key, item in command.items():
                if key in ["TimeStep", "tend", "atol", "rtol"]:
                    setattr(self, key, item)
                elif key == "ntr":
                    self.IncreaseNtr(item)
        
        #Check command priority. Finish command takes priority over repeat, takes priority over proceed
        for primarycommand in ["finish", "repeat", "proceed"]:
            if primarycommand in commands["primary"]:
                return eventdic, primarycommand

        #if no primarycommand was passed, return "finish"
        return eventdic, "finish"
    
    #Stays in Solver --> returns Solution
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
    
        
    def IncreaseNtr(self, val=10):
        self.ntr+=val
        print(f"Increasing ntr by {val} to {self.ntr}.")
        return