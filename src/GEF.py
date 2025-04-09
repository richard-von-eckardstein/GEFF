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
                                    model.UpdateVals, model.TimeStep, model.Initialise, model.events, inivals
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

    def GEFAlogrithm(self, reachNend=True, ensureConvergence=True, maxattempts=5):
        if reachNend: Nend=60 #set default Nend

        attempts=1
        while not(self.completed) and attempts<=maxattempts:
            t0 = 0
            vals = deepcopy(self.Solver.iniVals)
            yini = self.Solver.Initialise(self.Solver.iniVals, self.Solver.ntr)
            try:
                sol = self.Solver.SolveGEF(t0, yini, vals, reachNend=reachNend)
                if reachNend and ensureConvergence:
                    Ninf = sol.events["End of inflation"]["N"][-1]
                    if np.log10(abs(Ninf-Nend)) < -1: 
                        self.completed=True
                    else:
                        print("To verify a consistent run, checking stability against increasing ntr.")
                        self.Solver.IncreaseNtr(5)
                        Nend = Ninf
                else:
                    self.completed=True
            except TruncationError:
                attempts+=1
                print("A truncation error occured")
                self.Solver.IncreaseNtr(10)
        if attempts>maxattempts:
            print(f"The run did not finish after {attempts} attempts. Check the output for more information.")
            raise RuntimeError
            
        return sol
    
    def ModeByModeCrossCheck(self, sol, nmodes=500):
        TempGEF = self.CopySystem()
        TempGEF.SetUnits(False)
        self.Solver.ParseArrToUnitSystem(sol.t, sol.y, TempGEF)

        MbM = ModeByMode(TempGEF, self.settings)
        teval, Neval, ks, Ap, dAp, Am, dAm = MbM.ComputeSpectrum(nmodes)

        time = Timer()
        time.start()
        MbMres = []
        for n in range(self.Solver.ntr):
            FnGEF =  sol.y[4+3*n:4+3*(n+1),:]
            FnMbM  = []
            for j, t in enumerate(teval):
                FnMbM.append(MbM.EBGnFromModes(Ap[:,j], Am[:,j], dAp[:,j], dAm[:,j], t, ks, n=n))
            MbMres.append(FnMbM)
        np.array(MbMres)
        time.stop()

        keys = ["E", "B", "G"]
        for i, key in enumerate(keys):
            Nerr = Neval[100:] #ignore first 10 e-folds
            spl = CubicSpline( TempGEF.N, getattr(TempGEF, key) )(Nerr)
            err = abs( (MbMres[100:,i]-spl)/spl )
            err = np.array([np.average(err[-10*(i+1):-10*i]) for i in range(len(Nerr)//10)])
            Nerr = np.array([np.average(Nerr[-10*(i+1):-10*i]) for i in range(len(Nerr)//10)])

        if (err > 0.1).any():
            return False
        else:
            return True
    

    def RunGEF(self, ntr, tend=120., atol=1e-20, rtol=1e-6, ensureConvergence=True,
                reachNend=True, printstats=False, maxattempts=5, ModeCrossCheck=True):
        
        
        if not(self.completed):
            
            self.Solver.ntr=ntr
            self.Solver.tend=tend
            self.Solver.atol=atol
            self.Solver.rtol=rtol

            sol = self.GEFAlogrithm(
                                    reachNend=reachNend,
                                     ensureConvergence=ensureConvergence,
                                    maxattempts=maxattempts
                                    )
            
            if printstats: PrintSolution(sol)
            if ModeCrossCheck:
                MbMagreement = self.ModeByModeCrossCheck(sol, nmodes=500)
                
                if (MbMagreement):
                    print("The GEF agrees with the Mode by Mode solution. Storing results in GEF-object.")
                    self.Solver.ParseArrToUnitSystem(sol.t, sol.y, self)
                else:
                    print("The GEF does not agree with the Mode by Mode solution.")
                
            return sol
        else:
            print("This run is already completed, access data using GEF.'key'.")
            return


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
        def __init__(self, UpdateVals, TimeStep, Initialise, events, iniVals):
            self.Initialise = Initialise
            self.iniVals = iniVals
            self.__UpdateVals = UpdateVals
            self.TimeStep = TimeStep
            self.Events = events
            self.completed=False
        
        def __ode(self, t, y, vals, atol=1e-20, rtol=1e-6):
            self.__UpdateVals(t, y, vals)
            dydt = self.TimeStep(t, y, vals, atol=atol, rtol=rtol)
            return dydt
        
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
                print(f"The run did not finish after {sol.attempts} attempts. Check the output for more information.")
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