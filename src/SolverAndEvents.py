import numpy as np
from scipy.integrate import solve_ivp
#ToDo: Messages names after events.

#Errors
class EventError(Exception):
    pass

def noconsequence(sol, occurance):
    return {}

class Event:
    #Terminal events can modify a solver and pass concrete orders to the solver. The orders are:
    ### finish: finalise the solver
    ### repeat: repeat the last iteration of the solver
    ### proceed: continue solving from the point of termination
    #Any event may pass the secondary order "update" to the solver. This can update the following attributes of the solver:
    ### __ODE__, __tend, __atol, __rtol, __method 
    
    #Terminal event consequences should distinguish the cases: termination on this event, termination not on this event, termination without event.
    def __init__(x, name, func, terminal, direction, consequence):
        x.name = name
        x.func = func
        
        x.terminal = terminal
        x.direction = direction
        x.setflags()

        x.occurance = 0
        x.consequence = consequence
        return
    
    def EventConsequence(x, sol, occurance):
        result = x.consequence(sol, occurance)
        if (x.terminal and x.occurance == 1):
            #Deactivate termination at this event for future runs
            x.terminal=False
            x.setflags()
            #Deactivate consequences for this event for future runs
            x.consequence = noconsequence
        return result

    def setflags(x):
        x.func.terminal = x.terminal
        x.func.direction = x.direction
        return


class Solver:
    def __init__(x, name, ODE, t0, yini, tend, atol, rtol, Events=[], method="RK45", CommandHierarchy=["finish", "repeat", "proceed"]):
        x.name = name
        x.ODE = ODE
        x.__t0 = t0
        x.__yini = yini
        x.tend = tend
        x.atol = atol
        x.rtol = rtol
        x.Events = Events
        x.method = method
        x.__sols = []

        x.__CommandHierarchy = CommandHierarchy
        x.done = False

        return
    
    def Solve(x):
        while not(x.done) and len(x.__sols) < 10:
            events = [y.func for y in x.Events]
            step = 10**(np.floor(np.log10(abs(x.tend - x.__t0)) -3))
            teval = np.arange(x.__t0, x.tend, step)
            print(x.tend)
            try:
                sol = solve_ivp(
                    x.ODE, [x.__t0, x.tend], x.__yini, t_eval=teval, method=x.method, events=events, atol=x.atol, rtol=x.rtol)
                assert sol.success
            except: raise RuntimeError
            else: 
                x.__sols.append(sol)
                commands = x.AssessEvents(sol)
                x.execute(commands)
                
        if not(x.done):
            print(f"Failed to solve after {len(x.__sols)} attempts")
        x.Finalise()
        return
    
    def Finalise(x):
        y = []
        t = []
        for sol in x.__sols:
            t.append(sol.t)
            y.append(sol.y)
        t = np.concatenate(t)
        y = np.concatenate(y, axis=1)
        x.solution = sol
        x.solution.t = t
        x.solution.y = y
        return
    
    def execute(x, commands):    
        if commands["primary"]=="finish": x.finish()
        elif commands["primary"]=="repeat": x.repeat()
        elif commands["primary"]=="proceed": x.proceed()

        updatelist = commands["secondary"]
        print(updatelist)
        for updatedic in updatelist:
            for key in updatedic:
                x.update(key, updatedic[key])
        return

    def finish(x):
        print("Finishing")
        x.done=True
        return

    def repeat(x):
        print("Repeating")
        x.__sols.pop()
        return

    def proceed(x):
        print("Proceeding")
        x.__t0 = x.__sols[-1].t[-1]
        x.__yini = x.__sols[-1].y[:,-1]
        return

    def update(x, attr, val):
        allowedattributes = ["ODE", "tend", "atol", "rtol"]
        if hasattr(x, attr):
            if attr in allowedattributes:
                setattr(x, attr, val)
                print(f"Updating {attr} to {val}")
            else: print(f"{attr} cannot be changed by update().")
        else: print(f"Solver does not have the attribute {attr}.")
        return
    
    def AssessEvents(x, sol):
        commands = {"primary":[], "secondary":[]}
        for i, y in enumerate(x.Events):
            y.occurance += len(sol.t_events[i])
            consequence = y.EventConsequence(sol, y.occurance>0)
            for key in consequence.keys(): commands[key].append(consequence[key])
        
        primarycommands = commands["primary"]
        for commandhierarchy in x.__CommandHierarchy:
            if commandhierarchy in primarycommands:
                commands["primary"]=commandhierarchy
                return commands
            
        #No primary commands passed by events
        commands["primary"] = "finish"
        return commands

            
            


    




