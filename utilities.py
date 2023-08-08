import numpy as np
from EoM import FriedmannEq, ComputeSigma, ComputeImprovedSigma
from scipy.interpolate import CubicSpline

def PlotComp(ax1, ax2, N, Y, Nref, Yref, col="k:", label=None, interp="lin"):
    ax1.plot(N, Y, col, label=label)
    ax1.set_xlabel(r"$N_e$")
    if interp=="cubic":
        spl = CubicSpline(N, Y)
        ax2.plot(Nref, abs(spl(Nref)/Yref-1), col)
    else:
        spl = np.interp(Nref, N, Y)
        ax2.plot(Nref, abs(spl/Yref-1), col)
    ax2.set_yscale("log")
    ax2.set_ylabel("rel. err.")
    ax2.set_xlabel(r"$N_e$")
    
    return

def GetPhysQuantities(sol, Iterm, omega, f, SE=None, units=True):
    N = sol.y[0,:]
    a = np.exp(N)
    phi = sol.y[1,:]
    dphidt = sol.y[2,:]
    kh = np.exp(sol.y[3,:])
    V = potential(f*phi)/(f*omega**2)
    print(SE)
    if SE==None:
        E = sol.y[4,:]
        B = sol.y[5,:]
        G = sol.y[6,:]
        H = np.sqrt(FriedmannEq(a, dphidt, V, E, B, 0., f, omega/f))
        xi = GetXi(dphidt, Iterm, a, H, 0.)
        if units:
            phi = f*phi
            dphidt = omega*f*phi
            kh = kh*omega
            V = V*f**2*omega**2
            E = E*omega**4
            B = B*omega**4
            G = G*omega**4
            H = H*omega
        return N, a, phi, dphidt, kh, E, B, G, V, H, xi
    else: 
        delta = sol.y[4,:]
        rhoChi = sol.y[5,:]
        E = sol.y[6,:]
        B = sol.y[7,:]
        G = sol.y[8,:]
        H = np.sqrt(FriedmannEq(a, dphidt, V, E, B, rhoChi, f, omega/f))
        if SE=="Reg":
            sigma = np.array([ComputeSigma(E[i], B[i], H[i], f, omega) for i in range(len(E))])
            xi = GetXi(dphidt, Iterm, a, H, 0.)
            if units:
                phi = f*phi
                dphidt = omega*f*phi
                kh = kh*omega
                V = V*f**2*omega**2
                rhoChi = rhoChi**4
                E = E*omega**4
                B = B*omega**4
                G = G*omega**4
                H = H*omega
                sigma = omega*sigma
            return N, a, phi, dphidt, kh, delta, rhoChi, E, B, G, V, H, xi, sigma
        if SE=="Impr":
            sigmaE = np.array([ComputeImprovedSigma(E[i], B[i], G[i], H[i], f, omega)[0] for i in range(len(E))])
            sigmaB = np.array([ComputeImprovedSigma(E[i], B[i], G[i], H[i], f, omega)[1] for i in range(len(E))])
            xi = GetXi(dphidt, Iterm, a, H, sigmaB)
            if units:
                phi = f*phi
                dphidt = omega*f*phi
                kh = kh*omega
                V = V*f**2*omega**2
                rhoChi = rhoChi**4
                E = E*omega**4
                B = B*omega**4
                G = G*omega**4
                H = H*omega
                sigmaE = omega*sigmaE
                sigmaB = omega*sigmaB
            return N, a, phi, dphidt, kh, delta, rhoChi, E, B, G, V, H, xi, sigmaE, sigmaB
        