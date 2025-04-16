import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from src.BGQuantities import DefaultQuantities
from src.BGQuantities.BGTypes import BGSystem
from src.Tools.ModeByMode import ModeByMode
from scipy.interpolate import CubicSpline
import importlib.util as util
import os
import warnings
from copy import deepcopy
from src.Tools.timer import Timer

class TruncationError(Exception):
    pass

def ModelLoader(modelname):
    modelpath = os.path.join("Models/", modelname+".py")
    #Check if Model exists
    try:
        #Load ModelAttributes from GEFFile
        spec = util.spec_from_file_location(modelname, modelpath)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except:
        raise FileNotFoundError(f"No model found under '{modelpath}'")
    
def PrintSolution(sol):
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

class GEF(BGSystem):
    """
    A class used to solve the GEF equations given a set of initial conditions. The class also comes with several useful utility functions.
    
    ...
    
    Attributes
    ----------
    
    beta : float
        Coupling strength of the inflaton to the gauge fields, I_2(phi) = beta/Mpl
    units : Boolean
        Are all quantities treated as dimensionful or dimensioneless?
    H0 : float
        initial value of the Hubble rate in the dimensional unit system. Used for unit conversion.
    Mpl : float
        value of the Planck mass in the dimensional unit system. Used for unit conversion.
    GEFData : None | str
        Path to file where GEF results are stored
    ModeData : None | str
        Path to file where Mode By Mode results are stored
        
    ...
    
    Methods
    -------
    
    ...
    """
 
    def __init__(
                self, model: str, beta: float, iniVals: dict, Funcs: dict,
                userSettings: dict = {}, GEFData: None|str = None, ModeData: None|str = None
                ):
        #...Documentation...
        #Get Model attributes
        model = ModelLoader(model)

        #Set GEF-name
        self.__name = model.name

        #Set coupling constant
        self.beta = beta
        
        #Define background quantities
        valuedic = self.__PrepareQuantities(model.modelQuantities, iniVals)
        #Define background functions
        functiondic = self.__PrepareFunctions(model.modelFunctions, Funcs)

        #Compute H0 from initial conditions
        rhoInf = 0.5*valuedic["dphi"]["value"]**2 + functiondic["V"]["func"](valuedic["phi"]["value"])
        rhoEM = (valuedic["E"]["value"] + valuedic["B"]["value"])/2
        rhoExtra = [rho(valuedic) for rho in model.modelRhos]
        H0 = np.sqrt( (rhoInf + rhoEM + sum(rhoExtra))/3 )
        MP = 1.

        #Define the GEFClass as a BGSystem using the background quantities, functions and unit conversions
        super().__init__(valuedic, functiondic, H0, MP)

        #Configure model settings
        self.__ConfigureModelSettings(model.modelSettings, userSettings)

        #Create the GEF solver class
        self.__SetupGEFSolver(model)

        #Add information about storage
        self.GEFData = GEFData
        self.ModeData = ModeData

        return
    
    def __str__(self):
        #Add the model information
        string = f"Model: {self.__name}, "
        #Add any additional settings if applicable
        if isinstance(self.settings, dict):
            for setting in self.settings.items():
                string += f"{setting[0]} : {setting[1]}, "
        #Add coupling strength
        string += f"beta={self.beta}"
        return string

    def __ConfigureModelSettings(self, modelSettings, userSettings):
        settings = {}
        #Check if user specified any model settings, if not, use default settings
        for setting in modelSettings.keys():
            try:
                settings[setting] = userSettings[setting]
            except:
                settings[setting] = modelSettings[setting]
        
        #pass settings-dictionary to class
        if settings == {}:
            self.settings = None
        else:
            self.settings = settings
        return
    
    def __SetupGEFSolver(self, model):
        inivals = self.CreateCopySystem()
        inivals.SetUnits(False)

        self.Solver = self.GEFSolver(
                                    model.UpdateVals, model.TimeStep, model.Initialise,
                                      model.events, inivals,
                                      MbMSettings={} # EDIT FOR SE
                                    )
        self.completed=False
        
        return
    
    def __PrepareQuantities(self, modelSpecific, iniVals):
        #Get default quantities which are always present in every GEF run
        spacetime = DefaultQuantities.spacetime
        inflaton = DefaultQuantities.inflaton
        gaugefield = DefaultQuantities.gaugefield
        auxiliary = DefaultQuantities.auxiliary
        
        #concatenate dictionary and update default values according to model
        quantities = deepcopy(spacetime | inflaton | gaugefield | auxiliary)
        quantities.update(deepcopy(modelSpecific))

        necessarykeys = []

        #initialise GEFValues
        for key, item in quantities.items(): 
            #Check if a complete GEF value requires knowledge of this value.
            item.setdefault("optional", True) #If the "optional flag is not set, it is assumed the key is optional"
            if not(item["optional"]):
                necessarykeys.append(key)
            #default key is not longer needed from here on out
            item.pop("optional")
    
            #Add initial data
            if key in iniVals.keys():
                #Initial data from input
                item["value"]=iniVals[key]
            else:
                try:
                    #initial data from default
                    item["value"] = item["default"]
                    item.pop("default")
                except:
                    item["value"] = None
                    if key in necessarykeys:
                        warnings.warn(f"No default value set for '{key}'")
                    
            self.__necessarykeys = necessarykeys
        return quantities
    
    def __PrepareFunctions(self, modelSpecific, Funcs):
        #Get default functions which are always present in every GEF run
        inflatonpotential = DefaultQuantities.inflatonpotential
        coupling = DefaultQuantities.coupling
        
        #concatenate dictionary of functions and update default functions according to model
        functions = deepcopy(inflatonpotential | coupling)
        functions.update(deepcopy(modelSpecific))

        functions["dI"]["func"] = lambda x: float(self.beta)
        functions["ddI"]["func"] = lambda x: 0.

        #initialise GEFFunctions
        for key, item in functions.items():
            if key in ["dI", "ddI"]:
                #inflaton--gauge field coupling initialised separetly
                func = item["func"]
            else:
                #Check if Function is passed by the User via Funcs
                try:
                    func = Funcs[key]
                except:
                    raise KeyError(f"'Funcs' needs to declare the function '{key}'")
            item["func"] = func
            #Define the function as a BGFunc

        return functions
    
    def PrintNecessaryKeys(self):
        print(f"Necessary keys for this GEF-setup are:\n{self.__necessarykeys}")


    def LoadGEFData(self):
        #Check if GEF has a file path associated with it
        if self.GEFData == None:
            print("You did not specify the file from which to load the GEF data. Set 'GEFData' to the file's path from which you want to load your data.")
            return
        else:
            #Check if file exists
            file = self.GEFData
            try:
                #Load from file
                input_df = pd.read_table(file, sep=",")
            except FileNotFoundError:
                raise FileNotFoundError(f"No file found under '{file}'")
        
        #Dictionary for easy access using keys

        data = dict(zip(input_df.columns[1:],input_df.values[:,1:].T))

        #Check if data file is complete
        for key in self.__necessarykeys:
            if key not in data.keys():
                raise KeyError(f"The file you provided does not contain information on the parameter'{key}'. Please provide a complete data file")

        #Befor starting to load, check that the file is compatible with the GEF setup.
        valuelist = self.ValuesList()
        for key in data.keys():
            if not(key in valuelist):
                raise AttributeError(f"The data table you tried to load contains an unkown value: '{key}'")
        
        #Store current units to switch back to later
        units=self.GetUnits()

        #GEF data is always stored untiless, thus it is assumed to be untiless when loaded.
        self.SetUnits(False)
        #Load data into background-value attributes
        for key, values in data.items():
            obj = getattr(self, key)
            obj.SetValue(values)
        self.SetUnits(units)
        self.completed=True

        return

    def SaveGEFData(self):
        if self.GEFData==None:
            print("You did not specify the file under which to store the GEF data. Set 'GEFData' to the location where you want to save your data.")
        else:
            path = self.GEFData

            #Create a dictionary used to create pandas data table
            dic = {}

            #remember the original units of the GEF
            units=self.GetUnits()

            #Data is always stored unitless
            self.SetUnits(False)

            valuelist = self.ValuesList()
            for key in valuelist:
                obj = getattr(self, key)
                #Make sure to not store unitialised BGVal instances
                if isinstance(obj.value, type(None)):
                    #If a necessary key is not initialised, the data cannot be stored. Unitialised optional keys are ignored.
                    if (key in self.__necessarykeys):
                        #restore original units before raising error
                        self.SetUnits(units)
                        raise ValueError(f"Incomplete data. No values assigned to '{key}'.")
                else:
                    #Add the quantities value to the dictionary
                    dic[key] = obj.value
            
            #Create pandas data frame and store the dictionary under the user-specified path
            output_df = pd.DataFrame(dic)  
            output_df.to_csv(path)

            #after storing data, restore original units
            self.SetUnits(units)
        return
    
    def EndOfInflation(x, tol=1e-4):
        pass

    class GEFSolver:
        def __init__(self, UpdateVals, TimeStep, Initialise, events, iniVals, MbMSettings):
            self.__Initialise = Initialise
            self.iniVals = iniVals
            self.InitialConditions = self.InitialiseFromSlowRoll
            
            self.__UpdateVals = UpdateVals
            self.TimeStep = TimeStep

            self.Events = events

            self.MbMSettings = MbMSettings

        def __ode(self, t, y, vals, atol=1e-20, rtol=1e-6):
            self.__UpdateVals(t, y, vals)
            dydt = self.TimeStep(t, y, vals, atol=atol, rtol=rtol)
            return dydt

        def RunGEF(self, ntr, tend, atol, rtol, nmodes=500, printstats=True, **AlgorithmKwargs):
            self.ntr=ntr
            self.tend=tend
            self.atol=atol
            self.rtol=rtol
            reachNend = AlgorithmKwargs.get("reachNend", True)
            ensureConvergence = AlgorithmKwargs.get("ensureConvergence", True)
            maxattempts = AlgorithmKwargs.get("maxattempts", 5)

            done=False
            attempt = 0
            while not(done) and attempt<3:
                attempt +=1
                sol, vals = self.GEFAlgorithm(reachNend, ensureConvergence, maxattempts) 

                if nmodes!=None:
                    print("Using last successful GEF solution to compute gauge-field mode functions.")
                    MbM = ModeByMode(vals, self.MbMSettings)
                    teval, Neval, ks, Ap, dAp, Am, dAm = MbM.ComputeModeSpectrum(nmodes)
                    
                    print("Performing mode-by-mode comparison with GEF results.")
                    agreement, ind = self.ModeByModeCrossCheck(MbM, vals, teval, Neval, ks, Ap, Am, dAp, dAm)

                    if agreement:
                       print(f"The mode-by-mode comparison indicates a convergent GEF run.")
                       done=True
                    else:
                        print(f"Attempting to solve GEF using self-correction at N={Neval[ind]}.")
                        self.InitialConditions = self.InitialiseFromMbM(
                                                                        sol, MbM, teval[ind], ks,
                                                                        {"Ap":Ap[:,ind], "dAp":dAp[:,ind],
                                                                         "Am":Am[:,ind], "dAm":dAm[:,ind]}
                                                                         )
                else:
                    done=True
            
            if attempt<3:
                print("GEF run successfully completed.")
                if printstats:
                    PrintSolution(sol)
                return sol
            else:
                print("GEF run was unsuccessful, returning last GEF solution.")

            return sol
        
        def GEFAlgorithm(self, reachNend=True, ensureConvergence=True, maxattempts=5):
            if reachNend: Nend=60 #set default Nend
            attempts=1
            done = False
            #Run GEF
            while not(done) and attempts<=maxattempts:
                try:
                    t0, yini, vals = self.InitialConditions()
                    sol = self.SolveGEF(t0, yini, vals, reachNend=reachNend)
                    if reachNend and ensureConvergence:
                        Ninf = sol.events["End of inflation"]["N"][-1]
                        if np.log10(abs(Ninf-Nend)) < -1: 
                            done=True
                        else:
                            print("To verify a consistent run, checking stability against increasing ntr.")
                            self.IncreaseNtr(5)
                            Nend = Ninf
                    else:
                        done=True
                except TruncationError:
                    attempts+=1
                    print("A truncation error occured")
                    self.IncreaseNtr(10)

            if attempts>maxattempts:
                print(f"The run did not finish after {attempts} attempts.")
                try:
                    sol
                    print("Proceeding with last successful solution.")
                    return sol, vals
                except:
                    raise RuntimeError(f"Not a single successful solution after {attempts} attempts.")
            
            self.ParseArrToUnitSystem(sol.t, sol.y, vals)
                
            return sol, vals
        
        def ModeByModeCrossCheck(self, MbM, vals, teval, Neval, ks, Ap, Am, dAp, dAm):
            FMbM = np.array([
                            MbM.EBGnFromModes(
                                        t, ks, Ap[:,i], Am[:,i], dAp[:,i], dAm[:,i], n=0,
                                        epsabs=self.atol, epsrel=self.rtol
                                            )[0]
                        for i, t in enumerate(teval)])


            keys = ["E", "B", "G"]
            errs = []

            Nerr = Neval[100:] #ignore first 10 e-folds
            l = len(Nerr)//10

            for i, key in enumerate(keys):
                spl = CubicSpline( vals.N, (vals.a/vals.kh)**4*getattr(vals, key) )(Nerr) #interpolate GEF solution
                #average error over 1 e-fold to dampen impact of short time-scale spikes
                errs.append( np.average( abs( (FMbM[100:,i]-spl) / spl )[-10*l:].reshape(l, 10), 1) )
            #Create e-fold bins of 1-efold corresponding to the error arrays in errs
            Nerr = np.average( Nerr[-10*l:].reshape(l, 10), 1)

            print("The mode-by-mode comparison finds the following relative deviations from the GEF solution:")
            for i, key in enumerate(keys):
                err = errs[i]
                errind = np.where(err == max(err))
                maxerr = np.round(100*err[errind][0], 1)
                Nmaxerr = np.round(Nerr[errind][0], 1)
                errend = np.round(100*err[-1], 1)
                Nerrend = np.round(Nerr[-1], 1)
                print(f"-- {key} --")
                print(f"maximum relative deviation: {maxerr}% at N={Nmaxerr}")
                print(f"final relative deviation: {errend}% at N={Nerrend}")

            lowerrinds = []
            agreement=True
            for err in errs:              
                if (err > 0.1).any():
                    agreement=False
                    #find where the error is above 5%, take the earliest occurance, reduce by 1
                    errInd = np.where(err > 0.05)[0][0]-1                
                else:
                    errInd = len(Nerr) #meaningless placeholder
                lowerrinds.append(errInd)

            N0 = Nerr[min(lowerrinds)]
            print(N0)

            ind = np.where(Neval <= N0)[0][-1]

            return agreement, ind
        
        def InitialiseFromSlowRoll(self):
            t0 = 0
            vals = deepcopy(self.iniVals)
            yini = self.__Initialise(vals, self.ntr)
            return t0, yini, vals
        
        def InitialiseFromMbM(self, sol, MbM, t, ks, modes):
            def NewInitialiser():
                ntr = self.ntr
                rtol = self.rtol

                #Create array of initial data at time t based on ODE-solution
                yini = np.zeros((sol.y.shape[0]))
                for i in range(sol.y.shape[0]):
                    yini[i] = CubicSpline(sol.t, sol.y[i,:])(t)

                #reiinitialise Temp using "Initialise" to zero out all GEF-bilinear values
                Temp = deepcopy(self.iniVals)
                self.ParseArrToUnitSystem(t, yini, Temp) 
                ytmp = self.__Initialise(Temp, ntr)
                gaugeinds = np.where(ytmp==0.)[0]

                # compute En, Bn, Gn, for n>1 from Modes
                yini[gaugeinds[3:]] = np.array(
                                        [MbM.EBGnFromModes(
                                                            t, ks,
                                                            modes["Ap"], modes["Am"],
                                                            modes["dAp"], modes["dAm"],
                                                            n=n,epsabs=1e-20, epsrel=rtol*1e-2
                                                            )[0]
                                        for n in range(1,ntr+1)]
                                        ).reshape(3*ntr)
                
                #Prepare value system for solver
                self.ParseArrToUnitSystem(t, yini, Temp)

                return t, yini, Temp
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
                except ValueError:
                    print(f"The run failed at t={vals.t}, N={vals.N}.")
                    raise TruncationError
                except RuntimeError:
                    print(f"The run failed at t={vals.t}, N={vals.N}.")
                    raise RuntimeError
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