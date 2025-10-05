"""
A module defining equations used to define a `GEFF.mbm.ModeSolver`.
"""
import numpy as np
from typing import Callable


def bd_classic(t: float, k : float) -> np.ndarray:
    r"""
    Returns gauge-field modes in Bunch&ndash;Davies vacuum:

    $$A_\lambda(k,t) \sim \frac{1}{\sqrt{2k}}exp{(-i \eta(t) k)}\, , \qquad -\eta(t) k \gg 1 \, .$$

    Parameters
    ----------
    t : float
        time of initialisation $t_{\rm init}$
    k : float
        comoving momentum $k$

    Returns
    -------
    y : NDArray
         $\sqrt{2k} A_\lambda(t_{\rm init},k)$ and $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$ in Bunch&ndash;Davies

    """
    return np.array([1., 0, 0, -1., 1, 0, 0, -1.])

def mode_equation_classic(t: float, y : np.ndarray, k : float, a : Callable,
                    xi : Callable, H : Callable) -> np.ndarray:
    r"""
    Mode equation for pure axion inflation:

    $$\ddot{A}_\lambda(t,k) + H \dot{A}_\lambda(t,k) + \left[\left(\frac{k}{a}\right)^2  - 2\lambda \left(\frac{k}{a}\right) \xi H \right]A_\lambda(t,k) = 0 \, .$$

    Parameters
    ----------
    t : float
        cosmic time $t$
    y : NDArray
        $\sqrt{2k} A_\lambda(t_{\rm init},k)$ and $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$
    k : float
        comoving momentum $k$
    a : Callable
        scale factor as function of time, $a(t)$
    xi : Callable
        instability parameter as function of time, $\xi(t)$
    H : Callable
        Hubble rate as function of time, $H(t)$
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y
    """
    
    dydt = np.zeros_like(y)
    
    dis1 = k / a(t)
    dis2 = 2*H(t)*xi(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = -(dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = -(dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = -(dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = -(dis1  - lam * dis2) * y[6]
    
    return dydt

def damped_bd(t : float, k : float, a : Callable, delta : Callable,
              sigmaE : Callable) -> np.ndarray:
    r"""
    Returns gauge-field modes in damped Bunch&ndash;Davies vaccum:

    $$A_\lambda(k,t) \sim \sqrt{\frac{\Delta(t)}{2k}}exp{(-i \eta(t) k)}\, , \qquad -\eta(t) k \gg 1 \, .$$

    Parameters
    ----------
    t : float
        time of initialisation $t_{\rm init}$
    k : float
        comoving momentum $k$
    a : Callable
        scale factor as function of time, $a(t)$
    sigmaE : Callable
        electric damping, $\sigma_{\rm E}(t)$
    delta : Callable
        cumulative electric damping $\Delta(t) = \exp{\left(-\int \sigma_{\rm E}(t) {\rm d}t\right)}$

    Returns
    -------
    y : NDArray
        $\sqrt{2k} A_\lambda(t_{\rm init},k)$ and $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$ in Bunch&ndash;Davies

    """
    yini = np.array([1., -1/2*sigmaE(t)*a(t)/k, 0, -1.,
                     1., -1/2*sigmaE(t)*a(t)/k, 0, -1.])*np.sqrt( delta(t) )
    return yini


def mode_equation_SE_no_scale(t: float, y : np.ndarray, k : float,
                    a : Callable, xieff : Callable, H : Callable,
                    sigmaE : Callable) -> np.ndarray:
    r"""
    Mode equation for scale-dependent fermionic axion inflation:

    $$\ddot{A}_\lambda(t, k)  + \big( H + \sigma_{\rm E} \big) \dot{A}_\lambda(t, k) + \left[ \left(\frac{k}{a}\right)^2 - \lambda \frac{k}{a} \big( 2 \xi H + \sigma_{\rm B} \big) \right] A_\lambda(t, k) = 0\, .$$

    Parameters
    ----------
    t : float
        cosmic time $t$
    y : NDArray
        $\sqrt{2k} A_\lambda(t_{\rm init},k)$ and $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$
    k : float
        comoving momentum $k$
    a : Callable
        scale factor as function of time, $a(t)$
    xieff : Callable
        effective instability parameter as function of time, $\xi_{\rm eff}(t)$
    H : Callable
        Hubble rate as function of time, $H(t)$
    sigmaE : Callable
        electric damping, $\sigma_{\rm E}(t)$
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y
    """
    
    dydt = np.zeros_like(y)
    
    drag = sigmaE(t)
    dis1 = k / a(t)
    dis2 = 2*H(t)*xieff(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = - drag * y[1] - (dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = - drag * y[3] - (dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = - drag * y[5] - (dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = - drag * y[7] - (dis1  - lam * dis2) * y[6]
    
    return dydt

def mode_equation_SE_scale(t: float, y : np.ndarray, k : float,
                    a : Callable, xi : Callable, H : Callable,
                    sigmaE : Callable, sigmaB : Callable,
                      kS : Callable) -> np.ndarray:
    r"""
    Mode equation for scale-independent fermionic axion inflation:

    $$\ddot{A}_\lambda(t, k)  + \big( H + \sigma_{\rm E} \Theta(t, k) \big) \dot{A}_\lambda(t, k) + \left[ \left(\frac{k}{a}\right)^2 - \lambda \frac{k}{a} \big( 2 \xi H + \sigma_{\rm B}\Theta(t, k) \big) \right] A_\lambda(t, k) = 0\, .$$

    Parameters
    ----------
    t : float
        cosmic time $t$
    y : NDArray
        $\sqrt{2k} A_\lambda(t_{\rm init},k)$ and $a\sqrt{2/k} \dot{A}_\lambda(t_{\rm init},k)$
    k : float
        comoving momentum $k$
    a : Callable
        scale factor as function of time, $a(t)$
    xi : Callable
        einstability parameter as function of time, $\xi(t)$
    H : Callable
        Hubble rate as function of time, $H(t)$
    sigmaE : Callable
        electric damping, $\sigma_{\rm E}(t)$
    sigmaB : Callable
        magnetic damping, $\sigma_{\rm B}(t)$
    kS : Callable
        no damping for $k > k_{\rm S}(t)$ from $\Theta(t, k)$
        
    Returns
    -------
    dydt : array
        an array of time derivatives of y
    """

    dydt = np.zeros_like(y)
    cut = max(a(t)*H(t), kS(t))
    theta = np.heaviside(cut - k, 0.5)
    
    drag = theta*sigmaE(t)
    dis1 = k / a(t)
    dis2 = 2*H(t)*xi(t) + theta*sigmaB(t)

    #positive helicity
    lam = 1.
    #Real Part
    dydt[0] = y[1]*dis1
    dydt[1] = - drag * y[1] - (dis1  - lam * dis2) * y[0]
    
    #Imaginary Part
    dydt[2] = y[3]*dis1
    dydt[3] = - drag * y[3] - (dis1  - lam * dis2) * y[2]
    
    #negative helicity
    lam = -1.
    #Real Part
    dydt[4] = y[5]*dis1
    dydt[5] = - drag * y[5] - (dis1  - lam * dis2) * y[4]
    
    #Imaginary Part
    dydt[6] = y[7]*dis1
    dydt[7] = - drag * y[7] - (dis1  - lam * dis2) * y[6]
    
    return dydt



