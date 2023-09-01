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
    ratio = omega/f
    if SE==None:
        E = sol.y[4,:]
        B = sol.y[5,:]
        G = sol.y[6,:]
        H = np.sqrt(FriedmannEq(a, dphidt, V, E, B, 0., f, ratio))
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
        H = np.sqrt(FriedmannEq(a, dphidt, V, E, B, rhoChi, f, ratio))
        if SE=="mix":
            sigmaE = np.array([ComputeImprovedSigma(E[i], B[i], G[i], H[i], ratio)[0] for i in range(len(E))])
            sigmaB = np.array([ComputeImprovedSigma(E[i], B[i], G[i], H[i], ratio)[1] for i in range(len(E))])
            xi = GetXi(dphidt, Iterm, a, H, 0.)
            if units:
                phi = f*phi
                dphidt = omega*f*dphidt
                kh = kh*omega
                V = V*f**2*omega**2
                rhoChi = rhoChi*omega**4
                E = E*omega**4
                B = B*omega**4
                G = G*omega**4
                H = H*omega
                sigmaE = omega*sigmaE
                sigmaB = omega*sigmaB
            return N, a, phi, dphidt, kh, delta, rhoChi, E, B, G, V, H, xi, sigmaE, sigmaB
        elif (-1. <= SE <=1.):
            sigmaE = np.array([ComputeSigmaCollinear(E[i], B[i], np.sign(G[i]), H[i], ratio, SE)[0] for i in range(len(E))])
            sigmaB = np.array([ComputeSigmaCollinear(E[i], B[i], np.sign(G[i]), H[i], ratio, SE)[1] for i in range(len(E))])
            xi = GetXi(dphidt, Iterm, a, H, 0.)
            if units:
                phi = f*phi
                dphidt = omega*f*dphidt
                kh = kh*omega
                V = V*f**2*omega**2
                rhoChi = rhoChi*omega**4
                E = E*omega**4
                B = B*omega**4
                G = G*omega**4
                H = H*omega
                sigmaE = omega*sigmaE
                sigmaB = omega*sigmaB
            return N, a, phi, dphidt, kh, delta, rhoChi, E, B, G, V, H, xi, sigmaE, sigmaB
        else:
            print(SE, "is not a valid choice for SE")
            return 
        
def EndOfInflation(t, a, H, tol=1e-6):
    N = np.log(a)
    n = int(len(t)/10)
    
    dHdt = (H[1:] - H[:-1])/(t[1:]-t[:-1])
    
    f = CubicSpline(N[n:-1], dHdt[n:]/H[n:-1]**2 + 1)
    
    delN = 1.
    RefineGrid = True
    r0 = 40.
    while RefineGrid:
        print(delN)
        x0 = np.arange(50, max(N[:-2]), delN)
        res = fsolve(f, x0, xtol=tol)
        N0 = res[0]
        print(res[0])
        fN0 = abs(f(N0))
        for r in res:
            fN1 = abs(f(r))
            if fN1 < fN0:
                fN0 = fN1
                N0 = r
        r1 = np.round(N0, 2)
        print(r1)
        if r1 == r0:
            RefineGrid=False
        else:
            r0 = r1
            delN = delN*0.5
            
    return N0, delN

def SaveMode(t, k, Ap, dAp, Am, dAm, af, name=None):
    logk = np.log10(k)
    N = list(np.log(af(t)))
    N = np.array([np.nan]+N)
    t = np.array([np.nan]+list(t))
    dic = {"t":t}
    dic = dict(dic, **{"N":N})
    for k in range(0, len(Ap[:,0])):
        dictmp = {"Ap_" + str(k) :np.array([logk[k]] + list(Ap[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"Am_" + str(k) :np.array([logk[k]] + list(dAp[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"dAp_" + str(k):np.array([logk[k]] + list(Am[k,:]))}
        dic = dict(dic, **dictmp)
        dictmp = {"dAm_" + str(k):np.array([logk[k]] + list(dAm[k,:]))}
        dic = dict(dic, **dictmp)
        
    if(name==None):
        filename = "Modes+Beta" + str(beta) + "+M6_16" +  + ".dat"
    else:
        filename = name

    DirName = os.getcwd()

    path = os.path.join(DirName, filename)

    output_df = pd.DataFrame(dic)  
    output_df.to_csv(path)
    
    return

def ReadModeFile(file):
    input_df = pd.read_table(file, sep=",")
    dataAp = input_df.values

    x = np.arange(3,dataAp.shape[1], 4)
    
    t = np.array(dataAp[1:,1])
    N = np.array(dataAp[1:,2])
    logk = np.array([(complex(dataAp[0,y])).real for y in x])
    Ap = np.array([[complex(dataAp[i+1,y]) for i in range(len(N))] for y in x])
    dAp = np.array([[complex(dataAp[i+1,y+1]) for i in range(len(N))] for y in x])
    Am = np.array([[complex(dataAp[i+1,y+2]) for i in range(len(N))] for y in x])
    dAm = np.array([[complex(dataAp[i+1,y+3]) for i in range(len(N))] for y in x])
    
    return t, N, logk, Ap, dAp, Am, dAm