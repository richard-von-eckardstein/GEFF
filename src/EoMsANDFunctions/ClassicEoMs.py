import numpy as np

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

def EoMF(vals, F, W, dlnkhdt):
    FE = F[:,0]
    FB = F[:,1]
    FG = F[:,2]

    kh = vals.kh
    a = vals.a
    scale = kh/a

    W[2,1] = -W[2,1]

    ntr = FE.shape[0]-1 #subtract 1 for index 0 

    bdrF = dlnkhdt*np.array([[(W[j,0] + (-1)**i*W[j,1]) for j in range(3)]
                                for i in range(ntr+1)]) / (4*np.pi**2)

    ScalarCpl = (vals.dI(vals.phi)*vals.dphi)

    dFdt = np.zeros(bdrF.shape)

    for n in range(ntr): #all bilinear up to ntr-1
        dFdt[n,0] = (bdrF[n, 0] - (4+n)*dlnkhdt*FE[n] - 2*scale*FG[n+1] + 2*ScalarCpl*FG[n])

        dFdt[n,1] = (bdrF[n, 1] - (4+n)*dlnkhdt*FB[n] + 2*scale*FG[n+1])

        dFdt[n,2] = (bdrF[n, 2] - (4+n)*dlnkhdt*FG[n] + scale*(FE[n+1] - FB[n+1]) + ScalarCpl*FB[n])

    #bilinears at truncation order ntr
    dFdt[-1,0] = (bdrF[-1,0] -  (4+ntr)*dlnkhdt*FE[-1] - 2*scale*FG[-2] + 2*ScalarCpl*FG[-1])

    dFdt[-1,1] = (bdrF[-1,1] - (4+ntr)*dlnkhdt*FB[-1] + 2*scale*FG[-2]) 

    dFdt[-1,2] = (bdrF[-1,2] - (4+ntr)*dlnkhdt*FG[-1] + scale*(FE[-2] - FB[-2]) + ScalarCpl*FB[-1])

    return dFdt


