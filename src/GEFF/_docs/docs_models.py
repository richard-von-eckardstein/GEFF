DOCS = {
    "define_units":r"""Define how the initial data is used to define th reference frequency and energy scale.
    
    * energy scale: $M_{\rm P}$ in Planck units
    * frequency scale: initial Hubble rate, $H$ (in Planck units)

    Parameters
    ----------
    input : dict
        the input passed by the user
    """,

    "initial_conditions":"""Define how to create an array of initial data to solve the ODE's.

    Parameters
    ----------
    vals :  BGSystem
        contains initial data
    ntr : int
        truncation number

    Returns
    -------
    yini : NDArray
        initial data in an array
    """,

    "update_values" : """Compute static variables.

    Parameters
    ----------
    t : float
        time
    y : float
        array of data
    vals :  BGSystem
        system of data storing static variables
    atol, rtol : float
        precisions (used for heaviside functions)
    """,

    "compute_timestep" : """Compute time derivatives for dynamical variables.

    Parameters
    ----------
    t : float
        time
    y : float
        array of data
    vals :  BGSystem
        system of data storing static variables
    atol, rtol : float
        precisions (used for heaviside functions)

    Returns
    -------
    dydt : NDArray
        time derivative of y
    """,

    "condition_EndOfInflation":"""
    Determine the end of inflation using $\ddot{a} = 0$

    Parameters
    ----------
    t : float
        time
    y : float
        array of data
    vals :  BGSystem
        system of data storing static variables
    """,

    "consequence_EndOfInflation":r"""
    Define how the solver response to a (non)-occurrence of the end of inflation.

    If the event occurs, the solultion is accepted. Else, increase $t_{\rm end}$ and continue solving.

    Parameters
    ----------
    vals :  BGSystem
        solution to an ODE
    occurrence : bool
        if the event has occurred during the ODE solution
    """,

    "condition_NegativeEnergies":r"""
    Determine if either $\langle {\bf E}^2 \rangle$ or $\langle {\bf B}^2 \rangle$ is negative.

    Parameters
    ----------
    t : float
        time
    y : float
        array of data
    vals :  BGSystem
        system of data storing static variables
    """
}