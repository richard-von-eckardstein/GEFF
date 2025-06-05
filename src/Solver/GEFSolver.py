import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

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
    def __init__(self, UpdateVals, TimeStep, Initialise, events, ModeByMode, iniVals):
        self.__Initialise = Initialise
        self.iniVals = iniVals
        self.InitialConditions = self.InitialiseFromSlowRoll
        
        self.__UpdateVals = UpdateVals
        self.TimeStep = TimeStep

        self.Events = events

        self.ModeByMode = ModeByMode

    def __ode(self, t, y, vals, atol=1e-20, rtol=1e-6):
        self.__UpdateVals(t, y, vals, atol=atol, rtol=rtol)
        dydt = self.TimeStep(t, y, vals, atol=atol, rtol=rtol)
        return dydt

    def RunGEF(self, ntr, tend, atol, rtol, nmodes=500, printstats=True, MbMattempts=5, **AlgorithmKwargs):
        self.ntr=ntr
        self.tend=tend
        self.atol=atol
        self.rtol=rtol
        reachNend = AlgorithmKwargs.get("reachNend", True)
        ensureConvergence = AlgorithmKwargs.get("ensureConvergence", False)
        maxattempts = AlgorithmKwargs.get("maxattempts", 5)

        done=False
        sol = None
        attempt=0
        while not(done) and attempt<MbMattempts:
            attempt +=1
            solnew, vals = self.GEFAlgorithm(reachNend, ensureConvergence, maxattempts)
            sol = self.UpdateSol(sol, solnew)
            self.ParseArrToUnitSystem(sol.t, sol.y, vals)

            if nmodes!=None:
                print("Using last successful GEF solution to compute gauge-field mode functions.")
                MbM = self.ModeByMode(vals)
                spec = MbM.ComputeModeSpectrum(nmodes)

                
                print("Performing mode-by-mode comparison with GEF results.")
                agreement, ReInitSpec = self.ModeByModeCrossCheck(MbM, spec)

                if agreement:
                    print(f"The mode-by-mode comparison indicates a convergent GEF run.")
                    done=True
                else:
                    Nreinit = np.round(ReInitSpec["N"], 1)

                    print(f"Attempting to solve GEF using self-correction starting from N={Nreinit}.")

                    self.InitialConditions = self.InitialiseFromMbM(sol, MbM, ReInitSpec)
            else:
                done=True
        
        if done:
            print("GEF run successfully completed.")
            if printstats:
                PrintSummary(sol)
            return sol
        else:
            raise RuntimeError(f"GEF did not complete after {attempt} attempts.")
    
    def GEFAlgorithm(self, reachNend=True, ensureConvergence=True, maxattempts=5):
        if reachNend: Nend=60 #set default Nend
        attempts=1
        done = False
        #Run GEF
        while not(done):
            try:
                t0, yini, vals = self.InitialConditions()
                sol = self.SolveGEF(t0, yini, vals, reachNend=reachNend)

                if reachNend and ensureConvergence:
                    Ninf = sol.events["End of inflation"]["N"][-1]
                    if abs(Ninf-Nend) < 0.1: 
                        done=True
                    else:
                        attempts+=1
                        if attempts > maxattempts:
                            break
                        print("To verify a consistent run, checking stability against increasing ntr.")
                        self.IncreaseNtr(5)
                        Nend = Ninf
                        
                else:
                    done=True
            except TruncationError:
                attempts+=1
                if attempts > maxattempts:
                    break
                print("A truncation error occured")
                self.IncreaseNtr(10)

        
        if attempts>maxattempts:
            print(f"The run did not finish after {maxattempts} attempts.")
            try:
                sol
                print("Proceeding with last successful solution.")
                return sol, vals
            except:
                raise RuntimeError(f"Not a single successful solution after {maxattempts} attempts.")

        return sol, vals
    
    def ModeByModeCrossCheck(self, MbM, spec):
        errs, Nerr = MbM.CompareToBackgroundSolution(spec, epsabs=1e-20, epsrel=1e-4)
        
        lowerrinds = []
        agreement=True
        for err in errs:              
            if (err > 0.05).any():
                agreement=False
                #find where the error is above 5%, take the earliest occurance, reduce by 1
                errInd = np.where(err > 0.025)[0][0]-1                
            else:
                errInd = len(Nerr)-1
            lowerrinds.append(errInd)

        N0 = Nerr[min(lowerrinds)]

        ind = np.where(spec["N"] <= N0)[0][-1]

        ReInitSlice = spec.TSlice(ind)

        return agreement, ReInitSlice
    
    def UpdateSol(self, solold, solnew):
        if solold==None:
            return solnew
        else:
            sol = solold
            indoverlap = np.where(solnew.t[0] > solold.t)[0][-1]
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
    
    def InitialiseFromSlowRoll(self):
        t0 = 0
        vals = deepcopy(self.iniVals)
        yini = self.__Initialise(vals, self.ntr)
        return t0, yini, vals
    
    def InitialiseFromMbM(self, sol, MbM, ReInitSpec):
        """def NewInitialiser():
            ntr = self.ntr
            rtol = self.rtol
            atol = self.atol

            treinit = ReInitSpec["t"]

            #Create unit system:
            Temp = deepcopy(self.iniVals)

            #Construct yini from interpolation:
            ytmp = np.array([(CubicSpline(sol.t, sol.y[i,:])(treinit)) for i in range(sol.y.shape[0])])

            #Parse yini to Temp
            self.ParseArrToUnitSystem(treinit, ytmp, Temp)

            #Use "Initialise" to zero out all GEF-bilinear values
            yini = self.__Initialise(Temp, ntr)
            gaugeinds = np.where(yini==0.)[0]

            #parse back E0, B0, G0 (assuming they are at the same spot, should be the case.)
            yini[gaugeinds[:3]] = ytmp[gaugeinds[:3]]

            # compute En, Bn, Gn, for n>1 from Modes
            yini[gaugeinds[3:]] = np.array(
                                    [
                                    MbM.IntegrateSpecSlice(ReInitSpec, n=n,epsabs=atol, epsrel=rtol*1e-2)[0]
                                    for n in range(1,ntr+1)
                                    ]
                                    ).reshape(3*ntr)
            
            self.ParseArrToUnitSystem(treinit, yini, Temp)

            return treinit, yini, Temp
        return NewInitialiser"""
        def NewInitialiser():
            ntr = self.ntr
            rtol = self.rtol
            atol = self.atol

            treinit = ReInitSpec["t"]

            #Use the original "Initialise" to zero out all GEF-bilinear values

            Temp = deepcopy(self.iniVals)

            Temp.N.value = ReInitSpec["N"]

            ytmp = self.__Initialise(Temp, ntr)

            gaugeinds = np.where(ytmp==0.)[0]


            #Append initial data at time t based on the original ODE-solution

            yini = np.zeros_like(ytmp)

            for i in range(len(ytmp)):

                if i not in gaugeinds[3:]:

                    yini[i] = (CubicSpline(sol.t, sol.y[i,:])(treinit))

            # compute En, Bn, Gn, for n>1 from Modes

            yini[gaugeinds[3:]] = np.array([
                                    MbM.IntegrateSpecSlice(ReInitSpec, n=n,epsabs=atol, epsrel=rtol*1e-2)[0]
                                    for n in range(1,ntr+1)
                                    ]
                                    ).reshape(3*ntr)



            #Prepare value system for solver

            self.ParseArrToUnitSystem(treinit, yini, Temp)



            return treinit, yini, Temp

        return NewInitialiser
        
    def ParseArrToUnitSystem(self, t, y, vals):
        ts = deepcopy(t)
        ys = deepcopy(y)
        vals.SetUnits(False)
        self.__UpdateVals(ts, ys, vals)
        return
    
    def SolveGEF(self, t0, yini, vals, reachNend=True):
        done = False
        attempts = 0
        sols = []

        eventfuncs = []
        eventdic = {}
        for event in self.Events:
            eventname = event.name
            if eventname == "End of inflation" and not(reachNend):
                print("Removing default event 'End of inflation'")
            else:
                eventfuncs.append(event.func)
                eventdic[eventname] = {"t":[], "N":[]}

        print(f"The solver aims at reaching t={self.tend} with ntr={self.ntr}.")
        while not(done) and attempts < 10:
            try:
                tend = self.tend
                atol = self.atol
                rtol = self.rtol

                teval = np.arange(np.ceil(10*t0), np.floor(10*tend) +1)/10 #hotfix

                sol = solve_ivp(self.__ode, [t0,tend], yini, t_eval=teval, args=(vals, atol, rtol),
                                method="RK45", atol=atol, rtol=rtol, events=eventfuncs)
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

                eventdic_new, command = self.__AssessEvents(sol.t_events, sol.y_events, vals, reachNend=reachNend)

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
    
    def __AssessEvents(self, tevents, yevents, vals, reachNend=True):
        commands = {"primary":[], "secondary":[]}
        eventdic = {}
        Events = deepcopy(self.Events)
        if not(reachNend):
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