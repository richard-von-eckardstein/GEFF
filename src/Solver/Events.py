
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

        x.consequence = consequence
        return
    
    def EventConsequence(x, sol, occurance):
        result = x.consequence(sol, occurance)
        return result

    def setflags(x):
        x.func.terminal = x.terminal
        x.func.direction = x.direction
        return




