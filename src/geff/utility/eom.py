"""
A module containing equations of motions for common quantities in GEF models.
"""
import numpy as np
import math
from typing import Tuple

def friedmann(*rhos) -> float:
    r"""
    Calculate the Hubble rate $H$ from the Friedmann equation.

    Parameters
    ----------
    rhos
        list of energy densities

    Returns
    -------
    H : float
        the Hubble rate
    """
    Hsq = (1/3) * (sum(rhos)) 
    return np.sqrt(Hsq)

def check_accelerated_expansion(rhos, ps):
    r"""
    Compute $6 M_{\rm P}^2 \ddot{a}/a$.

    Parameters
    ----------
    rhos : list
        energy densities
    ps : list
        pressures
    """
    return -(sum(rhos) + 3*sum(ps))


def klein_gordon(dphi : float, dV : float, H : float, friction : float) -> float:
    r"""
    Calculate the Klein&ndash;Gordon equation (including gauge-field friction).

    Parameters
    ----------
    dphi : float
        the inflaton velocity, $\dot{\varphi}$
    dV : float
        the inflaton potential gradient, $V_{,\varphi}$
    H : float
        the Hubble rate, $H$
    friction : float
        friction term, (e.g., $\beta/M_{\rm P} \langle \boldsymbol{E}\cdot \boldsymbol{B} \rangle$)

    Returns
    -------
    ddphi : float
        the inflaton acceleration $\ddot{\varphi}$
    """
    return (-3*H * dphi - dV + friction)

def dlnkh(kh : float, dphi : float, ddphi : float,
             dI : float, ddI : float, xi : float,
               a : float, H : float) -> float:
    r"""
    Calculate the ${\rm d} \log k_{\rm h} / {\rm d}t$.

    Parameters
    ----------
    kh : float
        the instability scale $k_{\rm h}$
    dphi : float
        the inflaton velocity, $\dot{\varphi}$
    ddphi : float
        the inflaton acceleration, $\ddot{\varphi}$
    dI : float
        the inflaton--gauge-field coupling, $I_{,\varphi}$
    ddI : float
        derivative of the inflaton--gauge-field coupling, $I_{,\varphi \varphi}$
    xi : float
        the instability parameter, $\xi$
    a : float
        the scale factor, $a$
    H : float
        the Hubble rate, $H$

    Returns
    -------
    dlnkh : float
         ${\rm d} \log k_{\rm h} / {\rm d}t$.
    """
    
    r = 2*abs(xi)
        
    fc = a * H * r
    
    dHdt = 0. #approximation (quasi de Sitter)
    xiprime = (-dHdt * xi + ( ddI*dphi**2 + dI*ddphi)/2)/H
    rprime = 2*np.sign(xi)*xiprime
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def gauge_field_ode(F : np.ndarray, a : float, kh : float, sclrCpl : float,
          W : np.ndarray, dlnkhdt : float, L : int=10) -> np.ndarray:
    r"""
    Calculate the derivative of
    
    $$\mathcal{F}_\mathcal{E}^{(n)} =  \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{E} \cdot \operatorname{rot}^n \boldsymbol{E}\rangle \,  ,$$
    $$\mathcal{F}_\mathcal{B}^{(n)} =  \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{B} \cdot \operatorname{rot}^n \boldsymbol{B}\rangle \, , $$
    $$\mathcal{F}_\mathcal{G}^{(n)} =  -\frac{a^4}{2 k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{E} \cdot \operatorname{rot}^n \boldsymbol{B} + \boldsymbol{B} \cdot \operatorname{rot}^n \boldsymbol{E}\rangle \, . $$
    

    Parameters
    ----------
    F : NDArray
        array [$\mathcal{F}_\mathcal{E}^{(n)}$, $\mathcal{F}_\mathcal{B}^{(n)}$, $\mathcal{F}_\mathcal{G}^{(n)}$] in shape (3,ntr)
    a : float
        the scale factor, $a$
    kh : float
        the instability scale $k_{\rm h}$
    sclrCpl : float
        the inflaton gauge-field coupling, $2H\xi$
    W : NDarray
        boundary terms, shape (3,2)
    dlnkhdt : float
        logarithmic derivative of the instability scale, ${\rm d} \log k_{\rm h} / {\rm d}t$
    L : int
        polynomial order for closing ode at ntr

    Returns
    -------
    dFdt : NDArray
        the time derivative of [$\mathcal{F}_\mathcal{E}^{(n)}$, $\mathcal{F}_\mathcal{B}^{(n)}$, $\mathcal{F}_\mathcal{G}^{(n)}$], shape (3,ntr)
    """
    
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    scale = kh/a

    W[2,1] = -W[2,1]

    ns = np.arange(0,  FE.shape[0])

    lams = (-1)**ns

    bdrF = (
            dlnkhdt / (4*np.pi**2) 
            * (np.tensordot(np.ones_like(lams), W[:,0], axes=0)
                + np.tensordot(lams, W[:,1], axes=0))
            )

    dFdt = np.zeros_like(bdrF)

    dFdt[:-1,0] = (bdrF[:-1,0] - (4+ns[:-1])*dlnkhdt*FE[:-1]
                    - 2*scale*FG[1:] + 2*sclrCpl*FG[:-1])
    dFdt[:-1,1] = (bdrF[:-1,1] - (4+ns[:-1])*dlnkhdt*FB[:-1]
                    + 2*scale*FG[1:])
    dFdt[:-1,2] = (bdrF[:-1,2] - (4+ns[:-1])*dlnkhdt*FG[:-1]
                    + scale*(FE[1:] - FB[1:]) + sclrCpl*FB[:-1])

    #truncation conditions:
    L = FE.shape[0]//5
    ls = np.arange(1, L+1, 1)
    facl = np.array([math.comb(L, j) for j in range(1,L+1)])
    FEtr = np.sum( (-1)**(ls-1) * facl * FE[-2*ls], axis=0 )
    FBtr = np.sum( (-1)**(ls-1) * facl * FB[-2*ls], axis=0 )
    FGtr = np.sum( (-1)**(ls-1) * facl * FG[-2*ls], axis=0 )

    dFdt[-1,0] = (bdrF[-1,0] -  (4+ns[-1])*dlnkhdt*FE[-1]
                   - 2*scale*FGtr + 2*sclrCpl*FG[-1])
    dFdt[-1,1] = (bdrF[-1,1] - (4+ns[-1])*dlnkhdt*FB[-1]
                   + 2*scale*FGtr) 
    dFdt[-1,2] = (bdrF[-1,2] - (4+ns[-1])*dlnkhdt*FG[-1]
                   + scale*(FEtr - FBtr) + sclrCpl*FB[-1])

    return dFdt

def dlnkh_schwinger(kh : float, dphi : float, ddphi : float,
                dI : float, ddI : float,xieff : float, s : float,
                    a : float, H : float) -> float:
    r"""
    Calculate the ${\rm d} \log k_{\rm h} / {\rm d}t$ in presence of conductivities.

    Parameters
    ----------
    kh : float
        the instability scale $k_{\rm h}$
    dphi : float
        the inflaton velocity, $\dot{\varphi}$
    ddphi : float
        the inflaton acceleration, $\ddot{\varphi}$
    dI : float
        the inflaton--gauge-field coupling, $I_{,\varphi}$
    ddI : float
        derivative of the inflaton--gauge-field coupling, $I_{,\varphi \varphi}$
    xieff : float
        the effective instability parameter, $\xi + \sigma_{\rm B}/(2H)$
    s : float or val
        the effective electric conductivity, $\sigma_{\rm E}/(2H)$
    a : float
        the scale factor, $a$
    H : float
        the Hubble rate, $H$

    Returns
    -------
    dlnkh : float
         ${\rm d} \log k_{\rm h} / {\rm d}t$.
    """
    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    
    fc = a * H * r

    #approximations
    dsigmaEdt = 0.
    dsigmaBdt = 0.
    dHdt = 0.#vals.vals["Hprime"]# #approximation  dHdt = alphaH**2  (slow-roll)

    xieffprime = (-dHdt * xieff + (ddI*dphi**2 + dI*ddphi + a*dsigmaBdt)/2)/H
    sEprime = (-dHdt * s + a*dsigmaEdt/2)/H
    rprime = ( (np.sign(xieff)+xieff/sqrtterm)*xieffprime
                 + sEprime*(s+1/2)/sqrtterm )
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def ddelta(delta : float, sigmaE : float) -> float:
    r"""
    Calculate the derivative of the cumulative electric damping, $\Delta = \exp \left(-\int \sigma_{\rm E} {\rm d} t\right)$.

    Parameters
    ----------
    delta : float
        cumulative electric damping, $\Delta$
    sigmaE : float
        electric conductivity, $\sigma_{\rm E}$

    Returns
    -------
    ddelta : float
        the time derivative of $\Delta$
    """
    return -delta*sigmaE

def drhoChi(rhoChi : float, E : float, G : float,
               sigmaE : float, sigmaB : float, H : float) -> float:
    r"""
    Calculate the derivative of the fermion energy density.

    Parameters
    ----------
    rhoChi : float
        the fermion energy density, $\rho_{\chi}$
    E : float
        the electric field expecation value, $\langle \boldsymbol{E}^2 \rangle$
    G : float
        the expectation value of $-\langle \boldsymbol{E} \cdot \boldsymbol{B} \rangle$
    sigmaE : float
        electric conductivity, $\sigma_{\rm E}$
    sigmaB : float
        magnetic conductivity, $\sigma_{\rm B}$
    H : float
        the Hubble rate, $H$

    Returns
    -------
    float
        the time derivative of rhoChi
    """
    return (sigmaE*E - sigmaB*G - 4*H*rhoChi)

def gauge_field_ode_schwinger(F : np.ndarray, a : float, kh : float, sclrCpl : float,
            sigmaE : float, delta : float, 
            W : np.ndarray, dlnkhdt : float, L : int=10) -> np.ndarray:
    r"""
    Calculate the derivative of
    
    $$\mathcal{F}_\mathcal{E}^{(n)} =  \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{E} \cdot \operatorname{rot}^n \boldsymbol{E}\rangle \,  ,$$
    $$\mathcal{F}_\mathcal{B}^{(n)} =  \frac{a^4}{k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{B} \cdot \operatorname{rot}^n \boldsymbol{B}\rangle \, , $$
    $$\mathcal{F}_\mathcal{G}^{(n)} =  -\frac{a^4}{2 k_{\mathrm{h}}^{n+4}}\langle \boldsymbol{E} \cdot \operatorname{rot}^n \boldsymbol{B} + \boldsymbol{B} \cdot \operatorname{rot}^n \boldsymbol{E}\rangle \, , $$

    in the presence of Schwinger conductivities.

    Parameters
    ----------
    F : NDArray
        array [$\mathcal{F}_\mathcal{E}^{(n)}$, $\mathcal{F}_\mathcal{B}^{(n)}$, $\mathcal{F}_\mathcal{G}^{(n)}$] in shape (3,ntr)
    a : float
        the scale factor, $a$
    kh : float
        the instability scale $k_{\rm h}$
    sclrCpl : float
        the inflaton gauge-field coupling, $2H\xi_{\rm eff}$
    delta : float
        cumulative electric damping, $\Delta$
    W : NDarray
        boundary terms, shape (3,2)
    dlnkhdt : float
        logarithmic derivative of the instability scale, ${\rm d} \log k_{\rm h} / {\rm d}t$
    L : int
        polynomial order for closing ode at ntr

    Returns
    -------
    dFdt : NDArray
        the time derivative of [$\mathcal{F}_\mathcal{E}^{(n)}$, $\mathcal{F}_\mathcal{B}^{(n)}$, $\mathcal{F}_\mathcal{G}^{(n)}$], shape (3,ntr)
    """
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    scale = kh/a

    W[2,1] = -W[2,1]

    ns = np.arange(0,  FE.shape[0])

    lams = (-1)**ns

    bdrF = (dlnkhdt*delta / (4*np.pi**2)
            * (np.tensordot(np.ones_like(lams), W[:,0], axes=0)
                + np.tensordot(lams, W[:,1], axes=0))
            )

    dFdt = np.zeros(bdrF.shape)

    dFdt[:-1,0] = (bdrF[:-1,0] - (4+ns[:-1])*dlnkhdt*FE[:-1]
                   - 2*sigmaE*FE[:-1] - 2*scale*FG[1:] + 2*sclrCpl*FG[:-1])
    dFdt[:-1,1] = (bdrF[:-1,1] - (4+ns[:-1])*dlnkhdt*FB[:-1]
                    + 2*scale*FG[1:])
    dFdt[:-1,2] = (bdrF[:-1,2] - (4+ns[:-1])*dlnkhdt*FG[:-1]
                   - sigmaE*FG[:-1] + scale*(FE[1:] - FB[1:]) + sclrCpl*FB[:-1])
    
    #truncation conditions:
    ls = np.arange(1, L+1, 1)
    facl = np.array([math.comb(L, j) for j in range(1,L+1)])
    FEtr = np.sum( (-1)**(ls-1) * facl * FE[-2*ls], axis=0 ) #-2*ls instead of -2*ls+1 since -1 is ntr not ntr-1
    FBtr = np.sum( (-1)**(ls-1) * facl * FB[-2*ls], axis=0 )
    FGtr = np.sum( (-1)**(ls-1) * facl * FG[-2*ls], axis=0 )

    #bilinears at truncation order ntr
    dFdt[-1,0] = (bdrF[-1,0] - (4+ns[-1])*dlnkhdt*FE[-1]
                   - 2*sigmaE*FE[-1] - 2*scale*FGtr + 2*sclrCpl*FG[-1])
    dFdt[-1,1] = (bdrF[-1,1] - (4+ns[-1])*dlnkhdt*FB[-1]
                   + 2*scale*FGtr) 
    dFdt[-1,2] = (bdrF[-1,2] - (4+ns[-1])*dlnkhdt*FG[-1]
                   - sigmaE*FG[-1] + scale*(FEtr - FBtr) + sclrCpl*FB[-1])

    return dFdt

def conductivities_collinear(a  : float, H  : float,
                           E  : float, B : float, G : float,
                             picture : int, omega : float) -> Tuple[float, float, float]:
    r"""
    Compute electric & magnetic conductivities and the damping scale $k_{\rm S}$
    assuming collinear E & M fields.

    Parameters
    ----------
    a : float
        the scale factor, $a$
    H : float
        the Hubble rate, $H$
    E : float
        the electric field expecation value, $\langle \boldsymbol{E}^2 \rangle$
    E : float
        the magnetic field expecation value, $\langle \boldsymbol{B}^2 \rangle$
    G : float
        the expectation value of $-\langle \boldsymbol{E} \cdot \boldsymbol{B} \rangle$
    picture : int
        an integer specifying electric (=-1) or magnetic pictures (=1) 
    omega : float
        the reference frequency to convert from numerical to physical units

    Returns
    -------
    float
        the electric damping, $\sigma_{\rm E}$
    float
        the magnetic damping, $\sigma_{\rm B}$
    float
        the damping scale, $k_{\rm S}$
    """     
    mu = (E+B)
    if mu<=0:
        return 0., 0., 1e-4*a*H
    else:
        mu = (mu/2)**(1/4)
        mz = 91.2/(2.43536e18)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*omega))))
        
        C = 41/12

        sigma = (C*gmu**3/(6*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B/E))))
        sigmaE =  np.sqrt(B) * (min(1., (1.- picture))*E + max(-picture, 0.)*B) * sigma / (E+B)         
        sigmaB = -np.sign(G) * np.sqrt(E)*(min(1., (1.+ picture))*B + max(picture,0.)*E)* sigma/(E+B)
        
        ks = C**(1/3)*gmu*E**(1/4)*a
        
        return sigmaE, sigmaB, ks
    
def conductivities_mixed(a  : float, H  : float,
                           E  : float, B : float, G : float,
                            omega : float) -> Tuple[float, float, float]:
    r"""
    Compute electric & magnetic conductivities and the damping scale $k_{\rm S}$
    in the mixed picture.

    Parameters
    ----------
    a : float
        the scale factor, $a$
    H : float
        the Hubble rate, $H$
    E : float
        the electric field expecation value, $\langle \boldsymbol{E}^2 \rangle$
    E : float
        the magnetic field expecation value, $\langle \boldsymbol{B}^2 \rangle$
    G : float
        the expectation value of $-\langle \boldsymbol{E} \cdot \boldsymbol{B} \rangle$
    omega : float
        the reference frequency to convert from numerical to physical units

    Returns
    -------
    float
        the electric damping, $\sigma_{\rm E}$
    float
        the magnetic damping, $\sigma_{\rm B}$
    float
        the damping scale, $k_{\rm S}$
    """     
    Sigma = np.sqrt((E - B)**2 + 4*G**2)
    if Sigma<=0:
        return 0., 0., 1e-4*a*H
    else:
        mz = 91.2/(2.43536e18)
        mu = ((Sigma)/2)**(1/4)
        gmz = 0.35
        gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*omega))))

        Eprime = np.sqrt( (E - B + Sigma)/2 )
        Bprime = np.sqrt( (B- E + Sigma)/2 )
        Sum = E + B + Sigma

        C = 41/12
        
        sigma = ( C*gmu**3/(6*np.pi**2) / (np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime)))
        
        ks = C**(1/3)*gmu*Eprime**(1/2)*a

        return 2**(1/2)*abs(G)*Eprime*sigma, -2**(1/2)*G*Bprime*sigma, ks