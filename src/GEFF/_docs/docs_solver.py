module_docs = """
    This module defines the internal ODE solver used by the GEF.

    The `BaseGEFSolver` is the the primary class supplied by this module. It defines an algorithm by which the equations of motion for a GEF model are solved.
    This makes use of the `GEFF.bgtypes.BGSystem` module to simplify conversions between numerical and physical units.

    While the `BaseGEFSolver`is solving the ODE's it can track one or multiple `Event` objects. These events correspond to certain conditions, for example, the end of inflation.
    Occurrences of these events can be designed to influence the solver. For example, an 'end of inflation' `Event` may demand that the solver reach the end of inflation.

    The `BaseGEFSolver` class can be used to configure customized solvers using the class factory `GEFSolver` to adapt it to a specific GEF model.
    """

DOCS = {
    "BaseGEFSolver":r"""
    A class used to solve the equations of motion defined by a GEF model.

    The main purpose of the class is to provide the `compute_GEF_solution` method to `GEFF.GEFClass.BaseGEF.run`. 
    Internally, the method uses `solve_eom`, which wraps `scipy.integrate.solve_ivp`.
    
    All specifications on the GEF model are encoded in the attributes `known_variables` and `known_events`, as well as
    the methods `vals_to_yini`, `update_vals` and `timestep`. 
    For illustration, purposes, these methods are configured such that the GEFSolver solves the
    EoMs for de Sitter expansion:
    $$\frac{{\rm d} \log a}{{\rm d} t} = H_0$$

    In practice, the class factory `GEFSolver` creates a subclass of `BaseGEFSolver` adapted to the EoMs of other models.
    """,

    "GEFSolver":r"""
    Create a subclass of `BaseGEFSolver` with custom equation of motions and initial conditions.

    The subclass is adjusted to the specific EoMs of a new GEF model by defining new methods for `BaseGEFSolver.vals_to_yini`, `BaseGEFSolver.update_vals` and `BaseGEFSolver.timestep` via `new_init`, `new_update_val` and `new_timestep`.
    Information about the classificiation of variables is encoded in `new_variables` which overwrites `BaseGEFSolver.known_variables`.
    
    The `new_init` needs to obey the following rules:
    1. The call signature and output matches `vals_to_yini`.
    2. The return array `yini` has a shape: #${\rm dynamical\, vars.} + 3(n_{\rm tr}+1) \times$#${\rm gauge\, fields}$.
    3. All dynamical variables are reflected in the first #${\rm dynamical\, vars.}$-indices of `yini`.
    4. All gauge variables are reflected in the last $3(n_{\rm tr}+1) \times$#${\rm gauge\, fields}$-indices of `yini`.

    The `new_update_vals` needs to obey the following rules:
    1. The call signature and output matches `update_vals`.
    2. It updates every static variable using values in `y` and `t`.

    The `new_timestep` needs to obey the following rules:
    1. The call signature and output matches `timestep`.
    2. It computes derivatives `dydt` for every dynamical or gauge varibale.
    3. Static variables from `new_update_vals` can be re-used, as it is called before `new_timestep`.
    4. The indices in `dydt` need to match those of `yini` returned by `new_init`.

    All these functions assume that `vals` is in numerical units throughout the computation.

    In addition, a new list of `Event` objects can be passed to the subclass using `new_events`

    For an example on how to define a new solver, see `GEFF.models.classic`.

    Parameters
    ----------
    new_init : Callable
        overwrites `BaseGEFSolver.vals_to_yini`
    new_update_vals : Callable
        overwrites `BaseGEFSolver.update_vals`
    new_timestep : Callable
        overwrites `BaseGEFSolver.timestep`
    new_events :  list of Event
        overwrites `BaseGEFSolver.known_events`
    new_variables :  dict
        overwrites `BaseGEFSolver.known_variables`

    Returns
    -------
    GEFSolver
        a subclass of BaseGEFSolver
    """,

    "Event":r"""
    An event which is tracked while solving the GEF equations.

    The class defines a function $f(t, y)$ which is used by `scipy.integrate.solve_ivp` to track occurrences of $f(t, y(t))=0$.
    The event can be `terminal` causing the solver to stop upon an event occurrence.
    The event only triggers if the event condition changes sign according to:
    - positive zero crossing: `direction=1`
    - negative derivative, `direction=-1` 
    - arbitrary zero crossing `direction=0`

    The zeros are recorded and returned as part of the solvers output.

    The function $f$ is encoded in the method `event_func` which is defined upon initialization.

    Within subclasses of `BaseGEFSolver` class, three subclasses of `Event` are used:
    1. `TerminalEvent`
    2. `ErrorEvent`
    3. `ObserverEvent` 
    """

}