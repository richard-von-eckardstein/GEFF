DOCS = {
    "module": r"""
    This module is used to compute and analyse spectra of gauge-field modes.

    Throughout this module, any gauge-field mode functions are represented by dimensionless variables

    $$\sqrt{2k} A_\lambda(t,k), \quad \sqrt{\frac{2}{k}}\, a(t)\dot{A}_\lambda(k, t), \quad \lambda = \pm 1$$

    where $A_\lambda(t,k)$ are the mode functions of helicity $\lambda$ and momentum $k$ for a canonically quantized Abelian gauge-field $A_\mu(t, \boldsymbol{x})$ in Coulomb & Weyl gauge.
    The momentum variables $k$ are always returned in numerical units, i.e., $\bar{k} = k/\omega$.

    The class `BaseModeSolver` is designed to solve the second order mode equation

    $$\ddot{A}_\lambda(t,k) + P_\lambda(t,k)\dot{A}_\lambda(t,k) + Q_\lambda(t,k)A_\lambda(t,k) = 0 \, .$$

    The base class in particular is set to solve the mode equation of pure axion inflation,
    $$P_\lambda(t,k) = H \, \qquad Q_\lambda(t,k) = \left(\frac{k}{a}\right)^2  - 2\lambda \left(\frac{k}{a}\right) \xi H \, ,$$
    with Hubble rate $H$, scale factor $a$ and instability parameter $\xi$.

    To create a mode solver with custom values for $P(t,k)$ and $Q(t,k)$, use the class factory `ModeSolver`.

    The module also contains a class `GaugeSpec` designed for directly working on the spectrum of modes $A_\lambda(t,k)$.
    In particular, it is used to integrate the spectrum to obtain the quantities

    $$ \mathcal{F}_\mathcal{E}^{(n)}(t) = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a^2 k^{n+3}}{2 \pi^2 k_{{\rm UV}}^{n+4}}  \sum_{\lambda}\lambda^n |\dot{A}_\lambda(t,k)|^2\,,$$
    $$ \mathcal{F}_\mathcal{G}^{(n)}(t) = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{a k^{n+4}}{2 \pi^2 k_{{\rm UV}}^{n+4}}\sum_{\lambda}\lambda^{n+1} \operatorname{Re}[\dot{A}_\lambda(t,k)A_\lambda^*(t,k)]\,,$$
    $$ \mathcal{F}_\mathcal{B}^{(n)}(t) = \int\limits_{0}^{k_{{\rm UV}}(t)}\frac{{\rm d} k}{k} \frac{k^{n+5}}{2 \pi^{2}k_{{\rm UV}}^{n+4}} \sum_{\lambda}\lambda^n |A_\lambda(t,k)|^2\,,$$

    which may be used to estimate the error of a GEF solution.
    """,

    "GaugeSpec":r"""
    A class representing a spectrum of gauge-field modes as a function of time.

    This class inherits from `dict` and needs the following keys:  
    't', 'N', 'k', 'Ap', 'dAp', 'Am', 'dAm'

    All quantities are represented in numerical units (see `geff.bgtypes.BGSystem`).

    The spectrum can be evaluated at certain times $t$ or for certain momenta $k$ by using `tslice` and `kslice`
    Furthermore, the spectrum contained in the object can be integrated to compute gauge-field expectation values.
     The result can be used to estimate the error of a GEF run.

    Attributes
    ----------
    t : NDArray
        the cosmic time coordinates $t$ of the spectrum
    N : NDArray
        the $e$-folds as a function of cosmic time, $N(t)$
    k : NDarray
        the momenta $k$ at which the spectrum is evaluated
    Ap, Am : NDarray
        the mode functions, $\sqrt{2 k} A_\pm(k, t)$
    dAp, dAm : NDarray
        the mode-function derivatives, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
    """,

    "GaugeSpec.estimate_GEF_error": r"""
        Estimate the relative deviation between a GEF solution and the mode spectrum by computing

        $$\varepsilon_\mathcal{X} = \left|1 - \frac{\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}}{\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}}\right|$$

        for $\mathcal{X} = \mathcal{E},\,\mathcal{B},\,\mathcal{G}$. Here, $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$ are the integrals computed by `integrate`, 
        $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ refer to the same quantity in `BG`.
        If the time coordinate of `BG` does not align with the spectrum, its values are interpolated.

        Because $k_{\rm UV}(t)$ increases monotonically, the spectrum contains only few relevant modes $k < k_{\rm UV}(t)$ at early times.
        This poses a problem for the numerical integration of $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$.
        To avoid claiming a disagreement between $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm MbM}$ and $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ due to this effect,
        errors with $\varepsilon_\mathcal{X} > \varepsilon_{\rm thr}$ are discarded until the first time when $\varepsilon_\mathcal{X} < \varepsilon_{\rm thr}$.

        As the integration result fluctuates significantly for few momenta $k < k_{\rm UV}(t)$ when using `simpson`,
        the errors can be binned by setting `binning`. The reported error is the average over a bin of width $(t_{i}, t_{i+\Delta})$ with $\Delta$ set by `binning`.
        This binned error is then associated to the time $(t_{i} + t_{i+\Delta})/2$. For `quad`, `binning` can also be set to `None`.
        For details on the integration methods `simpson` and `quad`, see `SpecSlice.integrate_slice`.

        Parameters
        ----------
        BG : BGSystem
            the system where $\big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ are stored
        references : list of str
            the names where $(k_{\rm UV}/a)^4 \big(\mathcal{F}_\mathcal{X}^{(0)}\big)_{\rm GEF}$ are stored in `BG`
        cutoff : str
            the name where the UV-cutoff, $k_{\rm UV}$, is stored in `BG`
        err_thr : float
            the error threshold $\varepsilon_{\rm thr}$
        binning : int or None
            the bin size $\Delta$ (no binning if `None`)
        verbose : bool
            if `True`, print a summary of the errors
        **IntegratorKwargs :  kwargs
            passed to `SpecSlice.integrate_slice`

        Returns
        -------
        errs : list of NDArray
            a list of the binned errors with entries $[\varepsilon_\mathcal{E},\varepsilon_\mathcal{B}, \varepsilon_\mathcal{G}]$
        terr : NDArray
            the time coordinates corresponding to `errs`
        og_errs : list of NDArray
            the same as `errs` but without binning
        """,

    "SpecSlice": r"""
    A class representing a spectrum of gauge-field modes at a time $t$.

    Instances of this class are created by `GaugeSpec.tslice`. The main purpose of this class is to integrate the spectrum at time $t$ using `integrate_slice`.

    Attributes
    ----------
    t : NDArray
        the cosmic time coordinates $t$ of the spectrum
    N : NDArray
        the $e$-folds as a function of cosmic time, $N(t)$
    k : NDarray
        the momenta $k$ at which the spectrum is evaluated
    Ap, Am : NDarray
        the mode functions, $\sqrt{2 k} A_\pm(k, t)$
    dAp, dAm : NDarray
        the mode-function derivatives, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
    """,

    "SpecSlice.integrate_slice":r"""
        Compute the three integrals $\mathcal{F}_\mathcal{X}^{(n)}(t)$ for $\mathcal{X} = \mathcal{E}, \mathcal{B},\mathcal{G}$ for a fixed time $t$ and index $n$.

        The integrals can either be computed directly using `simpson` or `quad` from `scipy.interpolate`. When using `quad` the data for $\sqrt{2 k} A_\pm(k, t)$, $\sqrt{2/k} \, e^{N(t)}\dot{A}_\pm(k, t)$
          are interpolated to obtain smooth functions. To avoid this, it is recommended to use `simpson`.

        When using `simpson`, the integral is only computed if $m > m_{\rm thr}$ momenta $k_i$ satisfy $k < k_{\rm UV}$. Otherwise, the integral is set to zero.

        When using `quad`, the absolute and relative tolerances of the integrator are set by `epsabs` and `epsrel`. The interpolation method is defined by `interpolator`.
        Currently, only `CubicSpline` and `PchipInterpolator` from `scipy.interpolate` are supported. The later is preferred as interpolating the oscillatory mode functions can be subject to "overshooting".
        See [scipy's tutorial for 1-D interpolation](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection) for more details.

        Parameters
        ----------
        n : int
            the integer $n$ in $\mathcal{F}_\mathcal{X}^{(n)}(t)$ for $\mathcal{X} = \mathcal{E}, \mathcal{B},\mathcal{G}$
        integrator : str
            set the integration method to `simpson` or `quad`
        modethr : int
            set $m_{\rm thr}$ when using `simpson`
        epsabs : float
            the absolute tolerance of `quad`
        epsrel : float
            the relative tolerance of `quad`  
        interpolator
            the interpolator used to get smooth functions for `quad`
        

        Returns
        -------
        FMbM : NDArray
            contains [$\mathcal{F}_\mathcal{E}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$, $\mathcal{F}_\mathcal{B}^{(n)}(t)$] and the error estimated by `quad`.
             When using `simpson` the error is a dummy output. The shape of the result is (3, 2) with the second index indicating the integral (at 0), or the error (at 1).
        """,

    "BaseModeSolver":r"""
    A class used to compute gauge-field modes evolving on a time-dependent background.

    This class is used to evolve the gauge-field modes $A_\pm(t,k)$ as defined by `mode_equation`.
     To do so, the evolution of the time-dependent background is obtained from a GEF solution.

    The modes are initialized deep inside the Bunch&ndash;Davies vacuum as given by `initialise_in_bd`. 
    Numerically, the initialization time is implicitly defined by the condition $k = 10^{5/2} k_{\rm UV}(t_{\rm ini})$, with $k_{\rm UV}(t)$ obtained from the GEF solution.
    At times $t < t_{\rm ini}$ the mode is assumed to be in Bunch&ndash;Davies.

    The mode equations are solved with an explicit Runge&ndash;Kutta of order 5(4), which is implemented in `scipy.integrate.solve_ivp`.

    For creating a custom subclass of `BaseModeSolver` with user-specified  mode equation and initial conditions, you can use the class factory `ModeSolver`.
    """,

    "ModeSolver":r"""
    Create a subclass of `BaseModeSolver` with custom mode equation and initial conditions.

    In case your GEF model does not follow the pre-defined gauge-field mode equation `BaseModeSolver.mode_equation`,
    or initial conditions, `BaseModeSolver.initialise_in_bd` this method  defines a new subclass with
     these methods replaced by `new_mode_eq` and `new_bd_init`.
    
    The method `new_mode_eq` needs to obey the following rules:
    1. The call signature is `f(t,y,k,**kwargs)`
    2. The arguments `t` / `k` expect floats representing time / momentum
    3. The argument `y` expects a `numpy.ndarrray` of shape (8,) with indices
        -  0 & 2 / 4 & 6: real & imaginary part of $\sqrt{2k} A_\lambda(t,k)$ for $\lambda = 1 \, / -1$
        -  1 & 3 / 5 & 7: real & imaginary part of $\sqrt{2/k}\, a \dot{A}_\lambda(t,k)$ for $\lambda = 1 \, / -1$
    4. The kwargs are functions of the argument `t`.
    5. The return is the time derivative of `y`

    The method `new_bd_init` needs to obey the following rules:
    1. The call signature is `f(t,k,**kwargs)`
    2. The arguments `t` / `k` expect floats representing time / momentum
    3. The kwargs are functions of the argument `t`.
    4. The return is a `numpy.ndarrray` of shape (8,)  with indices
        -  0 & 2 / 4 & 6: real & imaginary part of $\sqrt{2k} A_\lambda(t_{\rm init},k)$ for $\lambda = 1 \, / -1$
        -  1 & 3 / 5 & 7: real & imaginary part of $\sqrt{2/k}\, a \dot{A}_\lambda(t_{\rm init},k)$ for $\lambda = 1 \, / -1$
    
    The lists `ode_keys` and `init_keys` are handled as follows:
    - `ode_keys` and `init_keys` need to contain the keys associated to the respective kwargs of `new_mode_eq` and `new_bd_init`.
    - These keys correspond to names of `geff.bgtypes.Variable` objects belonging to a `geff.bgtypes.BGSystem` passed to the class upon initialisation.
        The respective `Variable` objects are interpolated to obtain functions of time. 
        These functions are then passed to to the corresponding keyword arguments of `new_mode_eq`  and `new_bd_init`.
    - `ode_keys` and `init_keys` are added to `BaseModeSolver.necessary_keys` of the new subclass.

    The `BaseModeSolver.cutoff` and `BaseModeSolver.atol` attributes can also be adjusted.

    Parameters
    ----------
    new_mode_eq : Callable
        a new mode equation
    ode_keys : list of str
        the non-standard keywords of `new_mode_eq`
    new_bd_init : Callable
        a new mode bd initial condition
    init_keys : list of str
        the non-standard keywords of `new_bd_init`
    new_cutoff : str
        the new `cutoff` attribute of the subclass
    default_atol : float
        the default absolute tolerance used by the subclass

    Returns
    -------
    NewModeSolver
        the newly defined subclass of `BaseModeSolver`

    Example
    -------
    ```python
        import numpy as np
        from geff.bgtypes import BGSystem, define_var, t, N, kh

        # Define a new mode equation:
        def custom_mode_eq(t, y, k, a, X, Y):
            #create a return array of the right shape
            dydt = np.ones_like(y)

            #compute real-part time derivatives for positive modes
            dydt[0] = k / a(t) * y[1] # a is a function of t.
            dydt[1] = X(t)/Y(t)*y[0] # X and Y are functions of t.

            #compute imag-part time derivatives for positive modes
            ...

            #compute real-part time derivatives for negative modes
            ...
            ...

            return dydt

        # Define a new initial condition for the modes:
        def custom_bd_init(t, k, alpha):
            y = alpha(t)*np.array([...]) # alpha is a function of t.
            return y

        # the kwargs of custom_mode_eq are 'a', 'X' and 'Y':
        custom_ode_keys = ['a', 'X', 'Y']

        # the kwarg of custom_bd_init is 'alpha':
        custom_init_keys = ['alpha']

        # Define the custom mode solver using the class factory:
        CustomModeSolver = ModeSolver(custom_mode_eq, custom_ode_keys,
                                         custom_bd_init, custom_init_keys)

        # To initialise CustomModeSolver we need a BGSystem. 
        # Its Variables need to have the right names however:
        # The default: 't', 'N', 'kh' were loaded from geff.bgtypes

        # Because of custom_mode_eq we also need 'a', 'X', 'Y'
        a = define_var("a", 0, 0)
        X = define_var("X", 2, 0)
        Y = define_var("Y", 2, 0)

        # For custom_bd_init we need 'alpha'
        alpha = define_var("alpha", 0, 0)

        # When in doubt, consult necessary_keys:
        print(CustomModeSolver.necessary_keys)

        # We create the BGSystem and initialise all its values:
        sys = BGSystem({t, N, kh, a, X, Y, alpha}, 1e-6, 1)
        sys.initialise("t")(...)
        ...

        # The values in sys can now be used to initialise CustomModeSolver
        MbM = CustomModeSolver(sys)

        # Let's compute a spectrum using the new setup:
        spec = MbM.compute_spectrum(100)
        ...
        ```
    """
}