import numpy as np
import math

def FriedmannSE(vals):
    Hsq = (1/3) * (0.5 * vals.dphi**2 + vals.V(vals.phi) + vals.H0**2*( 0.5*(vals.E+vals.B) + vals.rhoChi) ) 
    return np.sqrt(Hsq)

def EoMlnkhSE(vals):
    a = vals.a
    H = vals.H
    kh = vals.kh
    xieff = vals.xieff
    s = vals.s

    sqrtterm = np.sqrt(xieff**2 + s**2 + s)
    r = (abs(xieff) + sqrtterm)
    
    fc = a * H * r

    #approximations
    dsigmaEdt = 0.
    dsigmaBdt = 0.
    dHdt = 0.#vals.vals["Hprime"]# #approximation  dHdt = alphaH**2  (slow-roll)

    xieffprime = (-dHdt * xieff + 
                      (vals.ddI(vals.phi)*vals.dphi**2 
                       + vals.dI(vals.phi)*vals.ddphi
                        + a*dsigmaBdt)/2
                     )/H
    sEprime = (-dHdt * s + a*dsigmaEdt/2)/H
    rprime = (np.sign(xieff)+xieff/sqrtterm)*xieffprime + sEprime*(s+1/2)/sqrtterm
    fcprime = H*fc + dHdt*a*r + a*H*rprime
                
    return fcprime/kh

def EoMDelta(vals):
    return -vals.sigmaE*vals.delta

def EoMrhoChi(vals):
    return (vals.sigmaE*vals.E - vals.sigmaB*vals.G - 4*vals.H*vals.rhoChi)


def EoMFSE(F, kh, a, sclrCpl, sigmaE, delta, W, dlnkhdt, L=10):
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
    facl = np.array([math.comb(L, l) for l in range(1,L+1)])
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


