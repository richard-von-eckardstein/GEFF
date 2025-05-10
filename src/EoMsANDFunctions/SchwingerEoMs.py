import numpy as np

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


def EoMFSE(vals, F, W, dlnkhdt):
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    kh = vals.kh
    a = vals.a
    scale = kh/a

    sE = vals.sigmaE
    sB = vals.sigmaB

    W[2,1] = -W[2,1]

    ntr = FE.shape[0]-1 #subtract 1 for index 0 

    bdrF = dlnkhdt*vals.delta*np.array([[(W[j,0] + (-1)**i*W[j,1]) for j in range(3)]
                                for i in range(ntr+1)]) / (4*np.pi**2)

    ScalarCpl = (vals.dI(vals.phi)*vals.dphi + sB)

    dFdt = np.zeros(bdrF.shape)

    for n in range(ntr): #all bilinear up to ntr-1
        dFdt[n,0] = (bdrF[n, 0] - (4+n)*dlnkhdt*FE[n] - 2*sE*FE[n] - 2*scale*FG[n+1] + 2*ScalarCpl*FG[n])

        dFdt[n,1] = (bdrF[n, 1] - (4+n)*dlnkhdt*FB[n] + 2*scale*FG[n+1])

        dFdt[n,2] = (bdrF[n, 2] - (4+n)*dlnkhdt*FG[n] - sE*FG[n] + scale*(FE[n+1] - FB[n+1]) + ScalarCpl*FB[n])

    #bilinears at truncation order ntr
    dFdt[-1,0] = (bdrF[-1,0] -  (4+ntr)*dlnkhdt*FE[-1] - 2*sE*FE[-1] - 2*scale*FG[-2] + 2*ScalarCpl*FG[-1])

    dFdt[-1,1] = (bdrF[-1,1] - (4+ntr)*dlnkhdt*FB[-1] + 2*scale*FG[-2]) 

    dFdt[-1,2] = (bdrF[-1,2] - (4+ntr)*dlnkhdt*FG[-1] - sE*FG[-1] + scale*(FE[-2] - FB[-2]) + ScalarCpl*FB[-1])

    return dFdt


