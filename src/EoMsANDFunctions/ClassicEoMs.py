import numpy as np
import math

def Friedmann(vals):
    Hsq = (1/3) * (0.5 * vals.dphi**2 + vals.V(vals.phi) + vals.H0**2*( 0.5*(vals.E+vals.B) ) ) 
    return np.sqrt(Hsq)

def EoMphi(vals):
    return (-3*vals.H * vals.dphi - vals.dV(vals.phi) - vals.dI(vals.phi)*vals.G*vals.H0**2)

def EoMlnkh(vals):
    a = vals.a
    H = vals.H
    kh = vals.kh
    xi = vals.xi

    r = 2*abs(vals.xi)
        
    fc = a * H * r
    
    dHdt = 0.#vals.vals["Hprime"]# #approximation  dHdt = alphaH**2  (slow-roll)
    xiprime = (-dHdt * xi + (vals.ddI(vals.phi)*vals.dphi**2 + vals.dI(vals.phi)*vals.ddphi)/2)/H
    rprime = 2*np.sign(xi)*xiprime
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def EoMF(vals, F, W, dlnkhdt, L=10):
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    kh = vals.kh
    a = vals.a
    scale = kh/a

    W[2,1] = -W[2,1]

    ns = np.arange(0,  FE.shape[0])

    lams = (-1)**ns

    bdrF = (
            dlnkhdt / (4*np.pi**2) 
            * (np.tensordot(np.ones_like(lams), W[:,0], axes=0)
                + np.tensordot(lams, W[:,1], axes=0))
            )

    ScalarCpl = (vals.dI(vals.phi)*vals.dphi)

    dFdt = np.zeros_like(bdrF)

    dFdt[:-1,0] = (bdrF[:-1,0] - (4+ns[:-1])*dlnkhdt*FE[:-1]
                    - 2*scale*FG[1:] + 2*ScalarCpl*FG[:-1])
    dFdt[:-1,1] = (bdrF[:-1,1] - (4+ns[:-1])*dlnkhdt*FB[:-1]
                    + 2*scale*FG[1:])
    dFdt[:-1,2] = (bdrF[:-1,2] - (4+ns[:-1])*dlnkhdt*FG[:-1]
                    + scale*(FE[1:] - FB[1:]) + ScalarCpl*FB[:-1])

    #truncation conditions:
    ls = np.arange(1, L+1, 1)
    facl = np.array([math.comb(L, l) for l in range(1,L+1)])
    FEtr = np.sum( (-1)**(ls-1) * facl * (scale)**(-2*ls) * FE[-2*ls+1], axis=0 )
    FBtr = np.sum( (-1)**(ls-1) * facl * (scale)**(-2*ls) * FB[-2*ls+1], axis=0 )
    FGtr = np.sum( (-1)**(ls-1) * facl * (scale)**(-2*ls) * FG[-2*ls+1], axis=0 )

    dFdt[-1,0] = (bdrF[-1,0] -  (4+ns[-1])*dlnkhdt*FE[-1]
                   - 2*scale*FGtr + 2*ScalarCpl*FG[-1])
    dFdt[-1,1] = (bdrF[-1,1] - (4+ns[-1])*dlnkhdt*FB[-1]
                   + 2*scale*FGtr) 
    dFdt[-1,2] = (bdrF[-1,2] - (4+ns[-1])*dlnkhdt*FG[-1]
                   + scale*(FEtr - FBtr) + ScalarCpl*FB[-1])

    return dFdt
