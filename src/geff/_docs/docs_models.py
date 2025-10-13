DOCS = {
    "define_units":r"""Define how initial data is used to set the reference frequency, $\omega$ and energy scale, $\mu$.
    
    * energy scale: $M_{\rm P}$ in Planck units
    * frequency scale: initial Hubble rate, $H_0$ (in Planck units)
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
    Determine if either $\langle \boldsymbol{E}^2 \rangle$ or $\langle \boldsymbol{B}^2 \rangle$ is negative.

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