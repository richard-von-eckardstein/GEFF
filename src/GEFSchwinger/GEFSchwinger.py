###Working Version for SEPicture=-1, includes Boundary Corrections and Damping Corrections (properly) for sigmaE and sigmaB
###Working Version for SEPicture=1 (very slow for 1 bdr corr)
###Working Version for SEPicture=mix, includes Boundary Corrections and Damping Corrections (properly) for sigmaE and sigmaB
###Alt Damp=1 properly implemented

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from src.Tools.timer import Timer
import math
from mpmath import whitw, whitm, gamma, mp

class TruncationError(Exception):
    pass

def AddEventFlags(name, terminal=True, direction=1, final=True):
    def setflags(func):
        func.name = name
        func.terminal = terminal
        func.direction = direction
        func.final = final
        return func
    return setflags

def PrintSol(sol):
    print("The run terminated with the following statistics:")
    for attr in sol.keys():
        if attr not in ["y", "t", "y_events", "t_events", "sol", "events"]:
            print(rf"{attr} : {getattr(sol, attr)}")
    try:
        events = sol.events
        if len(events.keys())==0:
            print("No events occured during the run")
        else:
            print("The following events occured during the run:")
            for event in events.keys():
                time = events[event]["t"]
                efold = events[event]["N"]
                print(rf"{event} at t={time} or N={efold}")
    except: return
    finally: return

class GEF:
    def __init__(x, beta: float, ini: dict, V, dV, SEPicture: int|str="mix", SEModel: str="KDep", GEFData: None|str=None, ModeData: None|str=None, approx: bool=True):
        x.units = True
        x.completed = False
        x.alpha = 0 # artifcat of previous attempt, may be reinstated at later times

        x.beta = beta

        x.V = V
        x.dV = dV

        x.ini = ini
        x.ini["rhoChi"] = 0.
        x.ini["delta"] = 1.
        x.ini["sigmaE"] = 0.
        x.ini["s"] = 0.
        x.ini["sigmaB"] = 0.

        x.GEFData = GEFData
        x.ModeData = ModeData

        x.SEModel = SEModel
        if x.SEModel=="Old":
            x.GaugePos = 6
            if(approx): x.Whittaker = x.WhittakerApprox_WithSE
            else: x.Whittaker=x.WhittakerExact
        elif x.SEModel=="Del1":
            x.GaugePos = 5
            if(approx): x.Whittaker = x.WhittakerApprox_NoSE
            else: x.Whittaker=x.WhittakerExact
        elif x.SEModel=="KDep":
            x.GaugePos = 5
            x.deltaf = x.ApproxDeltaf
            if approx:
                x.WhittakerNoFerm = x.WhittakerApprox_NoSE
                x.WhittakerWithFerm = x.WhittakerApprox_WithSE
            else:
                x.WhittakerNoFerm = x.WhittakerExact
                x.WhittakerWithFerm = x.WhittakerExact
            x.EoMlnkS = x.EoMlnkSMixed
        else:
            x.GaugePos = 4
            if(approx): x.Whittaker = x.WhittakerApprox_NoSE
            else: x.Whittaker = x.WhittakerExact

        x.SEPicture = SEPicture
        if (SEPicture=="mix"):
            x.conductivity = x.ComputeImprovedSigma
        elif (-1 <= SEPicture <=1):
            x.conductivity = x.ComputeSigmaCollinear
            if SEPicture==1 and approx:
                x.Whittaker = x.WhittakerApprox_NoSE
                x.WhittakerWithFerm = x.WhittakerApprox_NoSE
                x.deltaf = lambda x: 1.
        else:
            x.conductivity = lambda y: 0., 0., 0.


        x.approx = approx
        x.omega = 1.
        x.f = 1.
        x.ratio = 1.
        #Need Unitful Potential once, to compute omega
        x.H0 = np.sqrt( ( 0.5*x.ini["dphi"]**2 + x.V(x.ini["phi"]) )/3 )
        x.Mpl = 1.
        x.Nend = 60
        return
    
    #Potentials and Couplings
    def potential(x):
        phi = x.f*x.vals["phi"]
        return x.V(phi) / (x.f*x.omega)**2
    
    def dVdphi(x):
        phi = x.f*x.vals["phi"]
        return x.dV(phi)/(x.f*x.omega**2)

    def dIdphi(x):
        dI = x.beta/x.f
        return dI*x.f

    def ddIddphi(x):
        ddI = 0.
        return ddI*x.f**2
    
    #Useful Quantities
    def FriedmannEq(x):
        Hsq = (1/3) * (0.5 * x.vals["dphi"]**2 + x.vals["a"]**(2*x.alpha)*
                       (x.potential() + x.ratio**2*(0.5*(x.vals["E"]+x.vals["B"]) + x.vals["rhoChi"])))
        return np.sqrt(Hsq)
    
    def GetXi(x):
        return (x.dIdphi() * x.vals["dphi"])/(2 * x.vals["H"])
    
    def GetS(x, sigma):
        return (x.vals["a"]**(x.alpha) * sigma)/(2* x.vals["H"])
            
    def ComputeSigmaCollinear(x):
        E0 = x.vals["E"]
        B0 = x.vals["B"]
        G0 = x.vals["G"]
        mu = (E0+B0)
        if mu<=0:
            return 0., 0., 1e-2*x.vals["kh"]
        else:
            mu = (mu/2)**(1/4)
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))
            
            H = x.vals["H"]
            a = x.vals["a"]

            frac = x.SEPicture
            sigma = ((a**x.alpha) * (41.*gmu**3/(72.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B0/E0)))))
            sigmaE =  np.sqrt(B0) * (min(1., (1.- frac))*E0 + max(-frac, 0.)*B0) * sigma / (E0+B0)         
            sigmaB = -np.sign(G0) * np.sqrt(E0)*(min(1., (1.+ frac))*B0 + max(frac,0.)*E0)* sigma/(E0+B0)
            
            ks = gmu**(1/2)*E0**(1/4)*a**(1-x.alpha)
            
            return sigmaE, sigmaB, ks

    def ComputeImprovedSigma(x):
        E0 = x.vals["E"]
        B0 = x.vals["B"]
        G0 = x.vals["G"]
        Sigma = np.sqrt((E0 - B0)**2 + 4*G0**2)
        if Sigma<=0:
            x.vals["sigmaE"] = 0.
            x.vals["sigmaB"] = 0.
            return 0., 0., 1e-2*x.vals["kh"]
        else:
            mz = 91.2/(2.43536e18)
            mu = ((Sigma)/2)**(1/4)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))

            Eprime = np.sqrt(E0 - B0 + Sigma)
            Bprime = np.sqrt(B0- E0 + Sigma)
            Sum = E0 + B0 + Sigma
            
            H = x.vals["H"]
            a = x.vals["a"]

            sigma = (a**x.alpha)*(41.*gmu**3/(72.*np.pi**2)
                     /(np.sqrt(Sigma*Sum)*H * np.tanh(np.pi*Bprime/Eprime)))
            
            ks = gmu**(1/2)*Eprime**(1/2)*a**(1-x.alpha)

            return abs(G0)*Eprime*sigma, -G0*Bprime*sigma, ks
            
    #Equations of Motions
    def EoMphi(x):
        alpha = x.alpha
        a = x.vals["a"]
        
        ddphiddt = ((alpha-3)*x.vals["H"] * x.vals["dphi"]
                - a**(2*alpha)*x.dVdphi() - a**(2*alpha)*x.dIdphi()*x.vals["G"]*x.ratio**2)
        return ddphiddt
    
    def EoMlnkh(x, ddphiddt, rtol=1e-6):
        alpha = x.alpha
        a = x.vals["a"]
        H = x.vals["H"]
        kh = x.vals["kh"]
        
        xieff = x.vals["xieff"]
        s = x.vals["s"]
        sb = xieff - x.vals["xi"]
        sqrtterm = np.sqrt(xieff**2 + s**2 + s)
        r = (abs(xieff) + sqrtterm)
        
        fc = a**(1-alpha) * H * r

        dsigmaEdt = 0.
        dsigmaBdt = 0.

        dHdt = alpha*H**2 #approximation  dHdt = alphaH**2  (slow-roll)
        xieffprime = (-dHdt * xieff + 
                      (x.ddIddphi()*x.vals["dphi"]**2 + x.dIdphi()*ddphiddt + a**(1-alpha)*(alpha*H*x.vals["sigmaB"] + dsigmaBdt))/2
                     )/H
        sEprime = (-dHdt * s + a**(1-alpha)*(alpha*H*x.vals["sigmaE"]+ dsigmaEdt)/2)/H
        rprime = (np.sign(xieff)+xieff/sqrtterm)*xieffprime + sEprime*(s+1/2)/sqrtterm
        fcprime = (1-alpha)*H*fc + dHdt*a**(1-alpha)*r + a**(1-alpha)*H*rprime
    
        #return np.heaviside(fcprime/kh, rtol)*np.heaviside((fc/kh-1+rtol), rtol)*fcprime/kh
        if (fcprime >= 0):
            if((kh-fc)/kh <=1e-3):
            #if ( (1-np.log(fc)/np.log(kh)) < rtol):
                dlnkhdt = fcprime/kh
            else:
                #print("c2",x.vals["t"])
                dlnkhdt = 0
        else:
            #print("c1",x.vals["t"])
            dlnkhdt = 0
    
        return dlnkhdt

    def EoMlnkSMixed(x, dEdt, dBdt, dGdt):
        alpha = x.alpha
        H = x.vals["H"]

        E0 = x.vals["E"]
        B0 = x.vals["B"]
        G0 = x.vals["G"]
        Sigma = np.sqrt((E0 - B0)**2 + 4*G0**2)
        if Sigma==0:
            return (1-alpha)*H
        mz = 91.2/(2.43536e18)
        mu = ((Sigma)/2)**(1/4)
        gmz = 0.35
        gmu2 = gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio)))

        Eprime2 = E0 - B0 + Sigma

        dlnkSdt = ((1-alpha)*H + 1/4*(dEdt-dBdt)/Sigma*(gmu2/4*41/(48*np.pi**2)*(E0-B0)/Sigma + 1)
                    + dGdt*G0/Sigma**2*(gmu2/4*41/(48*np.pi**2) + Sigma/Eprime2))
        return dlnkSdt
        
    def EoMlnkSFrac(x, dEdt, dBdt, dGdt):
        alpha = x.alpha
        H = x.vals["H"]

        E0 = x.vals["E"]
        B0 = x.vals["B"]
        mu = (E0+B0)
        if mu<=0:
            return (1-alpha)*H
        mu = (mu/2)**(1/4)
        mz = 91.2/(2.43536e18)
        gmz = 0.35
        gmu2 = gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio)))

        dlnkSdt = ((1-alpha)*H + 0.25*dEdt*(gmu2/8*41/(48*np.pi**2)/mu**4 + 1/E0 )
                    + dBdt*gmu2/32*41/(48*np.pi**2)/mu**4)
        return dlnkSdt
    
    def ApproxDeltaf(x, t):
        return np.exp(-x.vals["sigmaE"]/x.vals["H"]*np.log(x.vals["kS"]/x.vals["kh"]))
    
    def EoMDelta(x):
        return -x.vals["a"]**(x.alpha)*x.vals["sigmaE"]*x.vals["delta"]
                
    def EoMrhoChi(x):    
        return x.vals["a"]**(x.alpha)*(x.vals["sigmaE"]*x.vals["E"]/x.vals["rhoChi"]
                                        - x.vals["sigmaB"]*x.vals["G"]/x.vals["rhoChi"]- 4*x.vals["H"])
        
    def EoMrhoChiBar(x):    
        return x.vals["a"]**(x.alpha)*(x.vals["sigmaE"]*x.vals["EBar"]
                                        - x.vals["sigmaB"]*x.vals["GBar"]- 4*x.vals["H"]*x.vals["rhoChi"])

    def EoMF(x, dlnkhdt):
        FE = x.vals["F"][:,0]
        FB = x.vals["F"][:,1]
        FG = x.vals["F"][:,2]

        aAlpha = x.vals["a"]**x.alpha

        sigmaE = x.vals["sigmaE"]
        sigmaB = x.vals["sigmaB"]

        kh = x.vals["kh"]
        a = x.vals["a"]
        scale = kh/a

        Whitt = x.Whittaker()

        Whitt[2,1] = -Whitt[2,1]

        bdrF = dlnkhdt*x.vals["delta"]*np.array([[(Whitt[j,0] + (-1)**i*Whitt[j,1]) for j in range(3)]
                                    for i in range(x.ntr)]) / (4*np.pi**2)
        
        ScalarCpl = (x.dIdphi()*x.vals["dphi"] + aAlpha*sigmaB)
        
        dFdt = np.zeros(bdrF.shape)

        for n in range(x.ntr-1):
            dFdt[n,0] = (bdrF[n, 0] - (4+n)*dlnkhdt*FE[n] - 2*aAlpha * sigmaE*FE[n] 
                             - 2*aAlpha*scale*FG[n+1] + 2*ScalarCpl*FG[n])
            dFdt[n,1] = (bdrF[n, 1] - (4+n)*dlnkhdt*FB[n] + 2*aAlpha*scale*FG[n+1])

            dFdt[n,2] = (bdrF[n, 2] - (4+n)*dlnkhdt*FG[n] - aAlpha*FG[n]*sigmaE
                             + aAlpha*scale*(FE[n+1] - FB[n+1]) + ScalarCpl*FB[n])

        dFdt[-1,0] = (bdrF[-1,0] -  (4+x.ntr-1)*dlnkhdt*FE[-1] - 2*aAlpha*FE[-1]*sigmaE
                            - 2*aAlpha*scale*FG[-2] + 2*ScalarCpl*FG[-1])

        dFdt[-1,1] = (bdrF[-1,1] - (4+x.ntr-1)*dlnkhdt*FB[-1] + 2*aAlpha*scale*FG[-2]) 

        dFdt[-1,2] = (bdrF[-1,2] - (4+x.ntr-1)*dlnkhdt*FG[-1] - aAlpha*FG[-1]*sigmaE
                             + aAlpha*scale*(FE[-2] - FB[-2]) + ScalarCpl*FB[-1])

        return dFdt

    def EoMFBar(x, dlnkhdt):
        FE = x.vals["F"][:,0]
        FB = x.vals["F"][:,1]
        FG = x.vals["F"][:,2]

        GE = x.vals["FBar"][:,0]
        GB = x.vals["FBar"][:,1]
        GG = x.vals["FBar"][:,2]

        aAlpha = x.vals["a"]**x.alpha

        sigmaE = x.vals["sigmaE"]
        sigmaB = x.vals["sigmaB"]

        kh = x.vals["kh"]
        a = x.vals["a"]
        scale = kh/a

        Whitt = x.Whittaker()

        Whitt[2,1] = -Whitt[2,1]

        bdrF = dlnkhdt*np.array([[(Whitt[j,0] + (-1)**i*Whitt[j,1]) for j in range(3)]
                                    for i in range(x.ntr)]) / (4*np.pi**2)
        
        ScalarCpl = (x.dIdphi()*x.vals["dphi"])
        
        dFdt = np.zeros(bdrF.shape)

        #Damped+Undamped Modes
        for n in range(x.ntr-1):
            dFdt[n,0] = (bdrF[n, 0] - (4+n)*dlnkhdt*FE[n] - 2*aAlpha * sigmaE*min(FE[n],GE[n]) 
                             - 2*aAlpha*scale*FG[n+1] + 2*ScalarCpl*FG[n] + 2*aAlpha*min(FG[n],GG[n]) *sigmaB)

            dFdt[n,1] = (bdrF[n, 1] - (4+n)*dlnkhdt*FB[n] + 2*aAlpha*scale*FG[n+1])

            dFdt[n,2] = (bdrF[n, 2] - (4+n)*dlnkhdt*FG[n] - aAlpha*min(FG[n],GG[n])*sigmaE
                             + aAlpha*scale*(FE[n+1] - FB[n+1]) + ScalarCpl*FB[n] + aAlpha*min(FB[n],GB[n])*sigmaB)

        dFdt[-1,0] = (bdrF[-1,0] -  (4+x.ntr-1)*dlnkhdt*FE[-1] - 2*aAlpha*sigmaE*min(FE[-1],GE[-1])
                            - 2*aAlpha*scale*FG[-2] + 2*ScalarCpl*FG[-1] + 2*aAlpha*min(FG[-1],GG[-1]) *sigmaB)

        dFdt[-1,1] = (bdrF[-1,1] - (4+x.ntr-1)*dlnkhdt*FB[-1] + 2*aAlpha*scale*FG[-2]) 

        dFdt[-1,2] = (bdrF[-1,2] - (4+x.ntr-1)*dlnkhdt*FG[-1] - aAlpha*min(FG[-1],GG[-1])*sigmaE
                             + aAlpha*scale*(FE[-2] - FB[-2]) + ScalarCpl*FB[-1] + aAlpha*min(FB[-1],GB[-1])*sigmaB)
        
        #Damped Modes only
        scaleS = x.vals["kS"]/a

        WhittBar = x.WhittakerExactkS()

        WhittBar[2,1] = -WhittBar[2,1]

        dlnkSdt = x.EoMlnkS(dFdt[0,0], dFdt[0,1], dFdt[0,2])
        bdrG = dlnkSdt*np.array([[(WhittBar[j,0] + (-1)**i*WhittBar[j,1]) for j in range(3)]
                                    for i in range(x.ntr)]) / (4*np.pi**2)
        
        dGdt = np.zeros(bdrG.shape)

        for n in range(x.ntr-1):
            dGdt[n,0] = (bdrG[n, 0] - (4+n)*dlnkSdt*GE[n] - 2*aAlpha * sigmaE*GE[n] 
                             - 2*aAlpha*scaleS*GG[n+1] + 2*ScalarCpl*GG[n] + 2*aAlpha*GG[n]*sigmaB)

            dGdt[n,1] = (bdrG[n, 1] - (4+n)*dlnkSdt*GB[n] + 2*aAlpha*scaleS*GG[n+1])

            dGdt[n,2] = (bdrG[n, 2] - (4+n)*dlnkSdt*GG[n] - aAlpha*GG[n]*sigmaE
                             + aAlpha*scaleS*(GE[n+1] - GB[n+1]) + ScalarCpl*GB[n] + aAlpha*GB[n]*sigmaB)

        dGdt[-1,0] = (bdrG[-1,0] -  (4+x.ntr-1)*dlnkSdt*GE[-1] - 2*aAlpha*GE[-1]*sigmaE
                            - 2*aAlpha*scaleS*GG[-2] + 2*ScalarCpl*GG[-1] + 2*aAlpha*GG[-1]*sigmaB)

        dGdt[-1,1] = (bdrG[-1,1] - (4+x.ntr-1)*dlnkSdt*GB[-1] + 2*aAlpha*scaleS*GG[-2]) 

        dGdt[-1,2] = (bdrG[-1,2] - (4+x.ntr-1)*dlnkSdt*GG[-1] - aAlpha*GG[-1]*sigmaE
                             + aAlpha*scaleS*(GE[-2] - GB[-2]) + ScalarCpl*GB[-1] + aAlpha*GB[-1]*sigmaB)

        dFtotdt = np.zeros((2*x.ntr,3))
        dFtotdt[:x.ntr,:] = dFdt
        dFtotdt[x.ntr:,:] = dGdt

        return dFtotdt
            
    #Run GEF
    def InitialiseGEF(x):
        if (x.SEPicture != None and x.SEModel == "KDep"):
            yini = np.zeros((2*x.ntr*3+x.GaugePos))
        else:
            yini = np.zeros((  x.ntr*3+x.GaugePos))

        #ini is always in Planck units
        x.vals = x.ini.copy()

        x.f = x.Mpl
        x.omega = x.H0
        x.ratio = x.omega/x.f
        x.units = False

        yini[0] = 0
        yini[1] = x.ini["phi"]/x.f
        yini[2] = x.ini["dphi"]/(x.f*x.omega)
        yini[3] = np.log(abs(yini[2]*x.dIdphi()))
        
        if (x.SEPicture != None):
            if (x.SEModel == "Del1"):
                x.Ferm2=0.
                yini[4] = -10#x.ini["rhoChi"]
            elif (x.SEModel == "KDep"):
                #x.FermionEntry = 1
                x.Ferm2=1
                yini[4] = -10#x.ini["rhoChi"]
                x.vals["kS"] = 1e-3*yini[3]
            else:
                yini[4] = x.ini["delta"]
                yini[5] = -10#x.ini["rhoChi"]
        
        return yini
    
    def TimeStep(x, t, y, rtol=1e-6):
        x.DefineDictionary(t, y)

        dydt = np.zeros(y.shape)

        dydt[0] = x.vals["H"]
        dydt[1] = x.vals["dphi"]
        dydt[2] = x.EoMphi()

        dlnkhdt = x.EoMlnkh(dydt[2], rtol=rtol)
        dydt[3] = dlnkhdt       
        
        if (x.SEPicture != None):
            if (x.SEModel == "Old"):
                dydt[4] = x.EoMDelta()
                dydt[5] = x.EoMrhoChi()
                dFdt = x.EoMF(dlnkhdt)
                dydt[x.GaugePos:] = dFdt.reshape(x.ntr*3)
            elif (x.SEModel == "Del1"):
                dydt[4] = x.EoMrhoChi()
                dFdt = x.EoMF(dlnkhdt)
                dydt[x.GaugePos:] = dFdt.reshape(x.ntr*3)
            elif (x.SEModel == "KDep"):
                dydt[4] = x.EoMrhoChiBar()
                if x.Ferm2 == 0:
                    dGdt = x.EoMF(dlnkhdt)
                    dFdt = np.zeros((2*x.ntr,3))
                    #After tferm, both EoMs are the same
                    dFdt[:x.ntr,:] = dGdt
                    dFdt[x.ntr:,:] = dGdt
                else:
                    #Before tferm, F[:x.ntr,:] are barless
                    #Before tferm, F[x.ntr:,:] have bars
                    dFdt = x.EoMFBar(dlnkhdt)
                dydt[x.GaugePos:] = dFdt.reshape(2*x.ntr*3)
        else:
            dydt[x.GaugePos:] = dFdt.reshape(x.ntr*3)
                
        return dydt
    
    def DefineDictionary(x, t, y, atol=1e-6):
        x.vals["t"] = t
        x.vals["N"] = y[0]
        
        x.vals["a"]= np.exp(y[0])
        
        x.vals["phi"] = y[1]
        x.vals["dphi"] = y[2]

        x.vals["kh"] = np.exp(y[3])

        F = y[x.GaugePos:x.GaugePos+3*x.ntr]
        x.vals["F"] = F.reshape(x.ntr, 3)

        x.vals["E"] = x.vals["F"][0,0]*np.exp(4*(y[3]-y[0]))
        x.vals["B"] = x.vals["F"][0,1]*np.exp(4*(y[3]-y[0]))
        x.vals["G"] = x.vals["F"][0,2]*np.exp(4*(y[3]-y[0]))
        
        if (x.SEPicture == None):
            s = 0.
            x.vals["H"] = x.FriedmannEq()
            x.vals["xi"] = x.GetXi()
            x.vals["xieff"] = x.vals["xi"]
        else:
            if (x.SEModel == "Del1"):
                x.vals["kS"] = x.vals["kh"]
                x.vals["rhoChi"] = np.exp(y[4])
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"], _ = x.conductivity()
                if np.log(x.vals["kh"]/(x.vals["a"]*x.vals["H"]))<0:
                    x.vals["sigmaE"] = 0.
                    x.vals["sigmaB"] = 0.
                x.vals["s"] = 0
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"]
            elif (x.SEModel == "KDep"):
                x.vals["rhoChi"] = y[4]
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"], x.vals["kS"] = x.conductivity()
                if np.log(x.vals["kS"]/(x.vals["a"]*x.vals["H"]))<0:
                    x.vals["sigmaE"] = 0.
                    x.vals["sigmaB"] = 0.

                G = y[x.GaugePos+3*x.ntr:]
                x.vals["FBar"] = G.reshape(x.ntr, 3)

                logkS = np.log(x.vals["kS"])

                x.vals["EBar"] = x.vals["FBar"][0,0]*np.exp(4*(logkS-y[0]))
                x.vals["BBar"] = x.vals["FBar"][0,1]*np.exp(4*(logkS-y[0]))
                x.vals["GBar"] = x.vals["FBar"][0,2]*np.exp(4*(logkS-y[0]))

                if (np.log(x.vals["kS"]/x.vals["kh"]) > 1e-3 or x.Ferm2==0):
                    x.Whittaker = x.WhittakerWithFerm
                    x.Ferm2 = 0
                    x.vals["delta"] = x.deltaf(t)
                else:
                    x.Whittaker = x.WhittakerNoFerm
                    x.Ferm2 = 1
                    x.vals["delta"] = 1.
                x.vals["s"] = x.GetS(x.vals["sigmaE"])*(1-x.Ferm2)
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])*(1-x.Ferm2)
            else:
                x.vals["delta"] = y[4]
                x.vals["rhoChi"] = np.exp(y[5])
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"], ks = x.conductivity()
                """if np.log(ks/(x.vals["a"]*x.vals["H"])) < atol:
                    x.vals["sigmaE"] = 0.
                    x.vals["sigmaB"] = 0."""
                x.vals["s"] = x.GetS(x.vals["sigmaE"])
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])    
    
        return

    @AddEventFlags(True, 1)
    def __EndOfInflation__(x, t, y):
        dphi = y[2]
        V = x.V(x.f*y[1])/(x.f*x.omega)**2
        rhoEB = 0.5*(y[x.GaugePos]+y[x.GaugePos+1])*x.ratio**2*np.exp(4*(y[3]-y[0]))
        rhoChi = y[x.GaugePos-1]*x.ratio**2
        val = np.log(abs((dphi**2 + rhoEB + rhoChi)/V))
        return val
    
    def IncreaseNtr(x, val=10):
        x.ntr+=val
        print(f"Increasing ntr by {val} to {x.ntr}.")
        return
    
    def SetupSolver(x, reachNend):
        events = []
        eventnames = []
        if reachNend: 
            events.append(x.__EndOfInflation__)
            eventnames.append(x.__EndOfInflation__.name)

        eventdic = dict(zip(eventnames, [{"t":[], "N":[]} for event in eventnames]))
        yini = x.InitialiseGEF()
        t0 = 0.
        return t0, yini, events, eventdic
    
    def ObtainSolution(x, t0, tend, yini, atol, rtol, events):
        ODE = lambda t, y: x.TimeStep(t, y, rtol)

        teval = np.arange(10*t0, 10*tend +1)/10

        sol = solve_ivp(ODE, [t0,tend], yini, t_eval=teval,
                                 method="RK45", atol=atol, rtol=rtol, events=events)
        return sol
    
    def SolveGEF(x, tend=120., atol=1e-20, rtol=1e-6, reachNend=True, Ntol=-1):
        mp.dps = 8
        t = Timer()

        print(f"The solver aims at reaching t={tend}")

        t0, yini, events, eventdic = x.SetupSolver(reachNend=True)

        t.start()

        done=False
        attempts = 0
        
        print(f"Attempting run with ntr={x.ntr}")
        while not(done) and attempts<10:
            attempts += 1
            try:
                sol = x.ObtainSolution(t0, tend, yini, atol, rtol, events)
                if not(sol.success):
                    print(f"The run failed at t={x.vals['t']}, N={x.vals['N']}.")
                    return sol, tend, x.Nend, "stop"
            except ValueError:
                print(f"The run failed at t={x.vals['t']}, N={x.vals['N']}.")
                raise TruncationError
            except RuntimeError:
                raise RuntimeError
            else:
                if reachNend:
                    try: 
                        Ninf = sol.y_events[0][0,0]
                    except:
                        t0=sol.t[-5]
                        yini = sol.y[:,-5]

                        if yini[0] > x.Nend: Ninc = 5
                        else: Ninc=x.Nend-yini[0]

                        tdiff = np.round(Ninc/x.vals["H"])
                        #round again, sometimes floats cause problems in t_span and t_eval.
                        tend  = np.round(tend + tdiff, 1)

                        print(rf"The end of inflation was not reached by the solver. Increasing tend by {tdiff} to {tend}.")
                    else:
                        done = True
                        for i, eventname in enumerate(eventdic.keys()):
                            eventdic[eventname]["t"].append(sol.t_events[i])
                            eventdic[eventname]["N"].append(sol.y_events[i][:,0])
                        print(f"The end of inflation was reached at t={np.round(sol.t_events[i][-1], 1)} and N={np.round(Ninf, abs(Ntol))}.")
                        if attempts > 1:
                            order="repeat"
                        else: order="proceed"
                else:
                    done=True
            
        t.stop()

        for eventname in (eventdic.keys()):
            eventdic[eventname]["t"] = np.round(np.concatenate(eventdic[eventname]["t"]), 1)
            eventdic[eventname]["N"] = np.round(np.concatenate(eventdic[eventname]["N"]), 3)

        sol.events = eventdic

        if attempts != 1 and not(done):
            print(f"The run did not finish after {sol.attempts} attempts. Check the output for more information.")
            raise RuntimeError

        return sol, tend, Ninf, order
    
    def RunGEF(x, ntr, tend=120., atol=1e-20, rtol=1e-6, reachNend=True, printstats=False, Ntol=-1, maxattempts=5):
        x.ntr = ntr+1
        if not(x.completed):
            finished= False
            attempts=1
            while not(finished) and attempts<=maxattempts:
                try:
                    sol, tend, Ninf, order = x.SolveGEF(tend, atol=atol, rtol=rtol, reachNend=reachNend, Ntol=Ntol)
                    if order=="repeat":
                        print("Multiple iterations where necessary to estimate tend. Repeating the same run to check consistency.")
                    elif order=="proceed":
                        if np.log10(abs(Ninf-x.Nend)) < Ntol: 
                            finished=True
                        else:
                            Ninc = abs(Ninf - x.Nend)
                            tdiff = np.round(Ninc/x.vals["H"])
                            tend = min(np.round(sol.t[-1] + tdiff, 1), tend)
                            print("To verify a consistent run, checking stability against increasing ntr.")
                            x.IncreaseNtr(5)
                        x.Nend = Ninf
                    elif order=="stop":
                        if printstats: PrintSol(sol)
                        x.WriteOutGEFResults(sol)
                        return sol

                except TruncationError:
                    attempts+=1
                    print("A truncation error occured")
                    x.IncreaseNtr(10)  
            if attempts>maxattempts:
                print(f"The run did not finish after {attempts} attempts. Check the output for more information.")
                raise RuntimeError
            if printstats: PrintSol(sol)
            x.WriteOutGEFResults(sol)
            x.completed = True
            return sol
        else:
            print("This run is already completed, access data using GEF.vals")
            return
    
    def WriteOutGEFResults(x, sol):
        t = sol.t
        y = sol.y
        parsold = list(x.vals.keys())
        parsold.remove("F")
        newpars = ["ddphi", "dlnkh"] #, "E1", "B1", "G1", "Edot", "Bdot", "Gdot", "EdotBdr", "BdotBdr", "GdotBdr"]
        pars = parsold + newpars
        res = dict(zip(pars, [[] for par in pars]))
        if (x.SEModel == "KDep"):
            #x.FermionEntry = 1
            res["sigmaEk"] = []
            res["sigmaBk"] = []
            res["sk"] = []
            res["xieffk"] = []
            x.Ferm2 = 1
            x.vals["kS"] = x.vals["kh"]*1e-3 
        for i in range(len(t)):
            x.DefineDictionary(t[i], y[:,i])
            if (x.SEModel == "KDep"):
                x.vals["delta"] = 1*x.Ferm2 + (1-x.Ferm2)*x.deltaf(x.vals["t"])
                x.vals["sigmaEk"] = (1.-x.Ferm2)*x.vals["sigmaE"]
                x.vals["sigmaBk"] = (1.-x.Ferm2)*x.vals["sigmaB"]
                x.vals["s"] = x.GetS(x.vals["sigmaE"])
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])
                x.vals["sk"] = x.GetS(x.vals["sigmaEk"])
                x.vals["xieffk"] = x.vals["xi"] + x.GetS(x.vals["sigmaBk"])
            elif (x.SEModel == "Del1"):
                x.vals["delta"] = 1.
                x.vals["s"] = x.GetS(x.vals["sigmaE"])
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])
            ddphi = x.EoMphi()
            res["ddphi"].append(ddphi)
            dlnkhdt = x.EoMlnkh(ddphi)
            res["dlnkh"].append(dlnkhdt)
            for par in parsold:
                res[par].append(x.vals[par])
        for par in pars:
            res[par] = np.array(res[par])
        x.vals = res
        return

    def CreateDeltaFunction(x):
        if (x.completed):
            if(x.SEModel != "KDep" or x.SEPicture == None):
                print("This only works for SE Runs with SEModel = KDep")
            else:
                ts = x.vals["t"]
                kh = x.vals["kh"]
                kS = x.vals["kS"]
                sE = x.vals["sigmaE"]
                xieff = x.vals["xieff"]
                xi = x.vals["xi"]
                a = x.vals["a"]
                H = x.vals["H"]
                
                sigmaf = CubicSpline(ts, sE)
                khf = CubicSpline(ts, kh)
                kSf = CubicSpline(ts, kS)
                etaf = CubicSpline(ts, -1/(a*H))
                delta = []
                eta = []
                tferms = [ts[0]]

                for i in range(ts.size):
                    f = lambda x: np.log(kSf(x)/khf(ts[i]))
                    tferm = fsolve(f, ts[i], xtol=1e-6)[0]
                    #print(ts[i], tferm)
                    x.vals["t"] = ts[i]
                    x.vals["xieff"] = xieff[i]
                    x.vals["xi"] = xi[i]
                    x.vals["s"] = a[i]**(x.alpha) * sE[i]/(2* H[i])
                    x.vals["kh"] = kh[i]
                    #print(x)
                    if (tferm<ts[i]):
                        eta.append(etaf(tferm))
                        delta.append(np.exp(-quad(sigmaf, tferm, ts[i])[0]))
                        tferms.append(tferm)
                    else:
                        eta.append(etaf(ts[i]))
                        delta.append(1.)
                        tferms.append(ts[i])

                x.deltaf = CubicSpline(ts, delta)
                x.vals["t"] = ts
                x.vals["xieff"] = xieff
                x.vals["xi"] = xi
                x.vals["s"] = a**(x.alpha) * sE/(2* H)
                x.vals["kh"] = kh
                x.vals["tferm"] = tferms[1:]
                #plt.plot(ts, np.array(eta))
                #print(o)
                x.etaf = CubicSpline(ts, np.array(eta))
                print("delta function created, access using x.deltaf")
        else:
            print("You first need to RunGEF")
        return
    
    ### RETHINK THIS FUNCTION!
    """def IterateGEF(x, t0=0., t1=120.):
        if (x.completed):
            x.Unitless()
            if (x.SEModel == "KDep"):
                #x.DeltaFunc = True
                x.CreateDeltaFunction()
                #x.Whittaker = x.Whittaker_Interp
                x.WhittakerWithFerm = x.Whittaker_PostFermionEntry
                x.completed = False
                pars = x.vals.keys()
                for par in pars:
                    x.vals[par] = x.vals[par][0]
                sol = x.RunGEF(t0, t1)
                return sol
            else:
                x.completed = False
                pars = x.vals.keys()
                for par in pars:
                    x.vals[par] = x.vals[par][0]
                sol = x.RunGEF(t0, t1)     
                return sol
        else:
            print("You first need to RunGEF")
            return"""
        
    def SaveData(x):
        if (x.completed):
            #x.Unitful()
            #Data is always stored without units
            x.Unitless()
            if x.GEFData==None:
                path = f"./Out/GEF_Beta{x.beta}_SE{x.SEPicture}_{x.SEModel}.dat"
            else:
                path = x.GEFData

            output_df = pd.DataFrame(x.vals)  
            output_df.to_csv(path)
        else:
            print("You need to RunGEF first")
        return
            
    def LoadData(x):
        if x.GEFData == None:
            print("You did not specify the file from which to load the GEF data. Set 'GEFData' to the file's path from which you want to load your data.")
            return
        else:
            file = x.GEFData
            try:
                input_df = pd.read_table(file, sep=",")
            except FileNotFoundError:
                print("This file does not exist")
                raise FileNotFoundError
            
            data = dict(zip(input_df.columns[1:],input_df.values[1:,1:].T))
            
            names = ["t", "phi", "dphi", "H", "a", "E", "B", "G", "rhoChi", "sigmaE", "sigmaB", "kh"]
            #Check if data file is in order:

            for name in names:
                if name not in data.keys():
                    print("The file you provided does not contain information on the parameter " + name + ". Please provide a complete data file")
                    print("A complete file contains information on the parameters:" + names)
                    raise ImportError

            """if ("kS" in data.keys()) and (x.SEModel in ["Old", "Del1"]):
                print(f"The file you are attempting to load seems to refer to a 'KDep' run. This is incompatible with your GEF setup. Please check your file path.")
                raise ImportError
            elif ("kS" not in data.keys()) and (x.SEModel=="KDep"):
                print(f"The file you are attempting to load is not 'KDep' run. This is incompatible with your GEF setup. Please check your file path.")
                raise ImportError"""
            #Since GEF data is always stored untiless, it is assumed to be untiless when loaded
            x.units = False
            x.omega = x.H0
            x.f = x.Mpl
            x.ratio = x.omega/x.f
            
            if len(data["t"]) == 1:
                print("It seems your table only contains one data point. This indicates a GEF run which is not yet executed. We suggest you initialise your run anew and use self.RunGEF")
                print("the completed-Flag is set to False")
                x.completed = False
                
            else:
                x.completed = True

            if not(hasattr(x, "vals")): x.vals = x.ini.copy()
            for key in data.keys():
                    x.vals[key] = data[key]

            return
            
    def Unitless(x):
        omega = x.H0
        f = x.Mpl
        if (not(x.completed)):
            print("You need to RunGEF or LoadGEF first")
            return
        if (x.units):
            x.vals["t"] = x.vals["t"]*omega
            #x.vals["tferm"] = x.vals["tferm"]*omega
            x.vals["phi"] = x.vals["phi"]/f
            x.vals["dphi"] = x.vals["dphi"]/(f*omega)
            x.vals["H"] = x.vals["H"]/(omega)
            x.vals["E"] = x.vals["E"]/(omega)**4
            x.vals["B"] = x.vals["B"]/(omega)**4
            x.vals["G"] = x.vals["G"]/(omega)**4
            x.vals["rhoChi"] = x.vals["rhoChi"]/(omega)**4
            x.vals["sigmaE"] = x.vals["sigmaE"]/omega
            x.vals["sigmaB"] = x.vals["sigmaB"]/omega
            x.vals["kh"] = x.vals["kh"]/omega
            if (x.SEModel == "KDep"):
                x.vals["kS"] = x.vals["kS"]/omega
                x.vals["sigmaEk"] = x.vals["sigmaEk"]/omega
                x.vals["sigmaBk"] = x.vals["sigmaBk"]/omega
                x.vals["EBar"] = x.vals["EBar"]/(omega)**4
                x.vals["BBar"] = x.vals["BBar"]/(omega)**4
                x.vals["GBar"] = x.vals["GBar"]/(omega)**4
            x.omega = omega
            x.f = f
            x.ratio = x.omega/x.f
            x.units = False
        else:
            print("Already Unitless")
        return
            
    def Unitful(x):
        omega = x.H0
        f = x.Mpl
        if (not(x.completed)):
            print("You need to RunGEF or LoadGEF first")
            return
        if (not(x.units)):
            x.vals["t"] = x.vals["t"]/omega
            #x.vals["tferm"] = x.vals["tferm"]/omega
            x.vals["phi"] = x.vals["phi"]*f
            x.vals["dphi"] = x.vals["dphi"]*(f*omega)
            x.vals["H"] = x.vals["H"]*(omega)
            x.vals["E"] = x.vals["E"]*(omega)**4
            x.vals["B"] = x.vals["B"]*(omega)**4
            x.vals["G"] = x.vals["G"]*(omega)**4
            x.vals["rhoChi"] = x.vals["rhoChi"]*(omega)**4
            x.vals["sigmaE"] = x.vals["sigmaE"]*omega
            x.vals["sigmaB"] = x.vals["sigmaB"]*omega
            x.vals["kh"] = x.vals["kh"]*omega
            if (x.SEModel == "KDep"):
                x.vals["kS"] = x.vals["kS"]*omega
                x.vals["sigmaEk"] = x.vals["sigmaEk"]*omega
                x.vals["sigmaBk"] = x.vals["sigmaBk"]*omega
                x.vals["EBar"] = x.vals["EBar"]*(omega)**4
                x.vals["BBar"] = x.vals["BBar"]*(omega)**4
                x.vals["GBar"] = x.vals["GBar"]*(omega)**4
            x.omega = 1.
            x.f = 1.
            x.ratio= 1.
            x.units = True
        else:
            print("Already Unitful")
        return
                
    #Whittaker Functions
    def WhittakerApprox_NoSE(x):
        xi = x.vals["xieff"]
        if (abs(xi) >= 3):
            #print("halo")
            Fterm = np.zeros((3, 2))
            sgnsort = int((1-np.sign(xi))/2)

            xi = abs(xi)
            g1 = math.gamma(2/3)**2
            g2 = math.gamma(1/3)**2
            t1 = (3/2)**(1/3)*g1/(np.pi*xi**(1/3))
            t2 = -np.sqrt(3)/(15*xi)
            t3 = (2/3)**(1/3)*g2/(100*np.pi*xi**(5/3))
            t4 = (3/2)**(1/3)*g1/(1575*np.pi*xi**(7/3))
            t5 = -27*np.sqrt(3)/(19250*xi**3)
            t6 = 359*(2/3)**(1/3)*g2/(866250*np.pi*xi**(11/3))
            t7 = 8209*(3/2)**(1/3)*g1/(13162500*np.pi*xi**(13/3))
            t8 = -690978*np.sqrt(3)/(1861234375*xi**5)
            t9 = 13943074*(2/3)**(1/3)*g2/(127566140625*np.pi*xi**(17/3))
            Fterm[0, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9

            t1 = 1
            t2 = -9/(2**(10)*xi**2)
            t3 = 2059/(2**(21)*xi**4)
            t4 = -448157/(2**31*xi**6)
            Fterm[0, 1-sgnsort] = np.sqrt(2)*(t1 + t2 + t3 + t4)

            t1 = (2/3)**(1/3)*g2*xi**(1/3)/(np.pi)
            t2 = 2*np.sqrt(3)/(35*xi)
            t3 = -4*(2/3)**(1/3)*g2/(225*np.pi*xi**(5/3))
            t4 = 9*(3/2)**(1/3)*g1/(1225*np.pi*xi**(7/3))
            t5 = 132*np.sqrt(3)/(56875*xi**3)
            t6 = -9511*(2/3)**(1/3)*g2/(5457375*np.pi*xi**(11/3))
            t7 = 1448*(3/2)**(1/3)*g1/(1990625*np.pi*xi**(13/3))
            t8 = 1187163*np.sqrt(3)/(1323765625*xi**5)
            t9 = -22862986*(2/3)**(1/3)*g2/(28465171875*np.pi*xi**(17/3))
            Fterm[1, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9

            t1 = 1
            t2 = 11/(2**(10)*xi**2)
            t3 = -2397/(2**(21)*xi**4)
            t4 = 508063/(2**31*xi**6)
            Fterm[1, 1-sgnsort] = 1/np.sqrt(2)*(t1 + t2 + t3 + t4)

            t1 = 1/np.sqrt(3)
            t2 = -(2/3)**(1/3)*g2/(10*np.pi*xi**(2/3))
            t3 = 3*(3/2)**(1/3)*g1/(35*np.pi*xi**(4/3))
            t4 = -np.sqrt(3)/(175*xi**2)
            t5 = -41*(2/3)**(1/3)*g2/(34650*np.pi*xi**(8/3))
            t6 = 10201*(3/2)**(1/3)*g1/(2388750*np.pi*xi**(10/3))
            t7 = -8787*np.sqrt(3)/(21896875*xi**4)
            t8 = -1927529*(2/3)**(1/3)*g2/(4638768750*np.pi*xi**(14/3))
            t9 = 585443081*(3/2)**(1/3)*g1/(393158390625*np.pi*xi**(16/3))
            t10 = -65977497*np.sqrt(3)/(495088343750*xi**6)
            Fterm[2, sgnsort] = t1+t2+t3+t4+t5+t6+t7+t8+t9+t10

            t1 = 1
            t2 = -67/(2**(10)*xi**2)
            t3 = 21543/(2**(21)*xi**4)
            t4 = -6003491/(2**31*xi**6)
            Fterm[2, 1-sgnsort] = -np.sqrt(2)/(32*xi)*(t1 + t2 + t3 + t4) 
        else:
            Fterm = x.WhittakerExact()
        return Fterm
    
    def WhittakerApprox_WithSE(x):
        xieff = x.vals["xieff"]
        if (abs(xieff) >= 4):
            #print("ola")
            Fterm = np.zeros((3, 2))
            sgnsort = int((1-np.sign(xieff))/2)

            s = x.vals["s"]
            xi = abs(xieff)
            r = xi + np.sqrt(xi**2 + s**2 + s)
            psi = 2*np.sqrt(xi**2 + s**2 + s)/r
            rpsi = (psi/r**2)**(1/3)
            spsi = 5*s/psi**(2/3)
            
            g1 = math.gamma(2/3)**2
            g2 = math.gamma(1/3)**2
            
            t1 = 3**(1/3)*g1/np.pi
            t2 = -2/(5*np.sqrt(3))*(1+spsi)
            t3 = g2/(3**(1/3)*25*np.pi)*(1+spsi)**2
            t4 = 3**(1/3)*4*g1/(1575)*(1-27*spsi)
            t5 = 4*np.sqrt(3)/(875)*(-27/11+2*spsi + spsi**2)
            Fterm[0, sgnsort] = (psi/r)**(1/3)*(t1 + t2*rpsi + t3*rpsi**2 + t4*rpsi**3 + t5*rpsi**4)

            t1 = 1
            t2 = s/(16*xi**2)*(3*xi-r)/r
            t3 = s**2/(4*xi*r)
            Fterm[0, 1-sgnsort] = 2*np.sqrt(xi/r)*(t1 + t2 + t3)

            t1 = g2/(3**(1/3)*np.pi)
            t2 = 4*np.sqrt(3)/35
            t3 = -16*g2/(3**(1/3)*225*np.pi)
            Fterm[1, sgnsort] = (r/psi)**(1/3)*(t1 + t2*rpsi**2 + t3*rpsi**3)

            Fterm[1, 1-sgnsort] = 0.5*(r/xi)**(1/2)

            t1 = 1/np.sqrt(3)
            t2 = -g2/(3**(1/3)*5*np.pi)*(1+spsi)
            t3 = 3**(1/3)*6*g1/(35*np.pi)
            t4 = -4*np.sqrt(3)/(175)*(1+spsi)
            Fterm[2, sgnsort] = t1 + t2*rpsi + t3*rpsi**2 + t4*rpsi**3

            Fterm[2, 1-sgnsort] = -((3*xi -r)/xi + 8*s)/(16*np.sqrt(xi*r))
        else:
            Fterm = x.WhittakerExact()
        return Fterm
    
    def WhittakerExact(x):
        #print("mahalo", end='\r')
        xieff = x.vals["xieff"]
        s = x.vals["s"]
        r = (abs(xieff) + np.sqrt(xieff**2 + s**2 + s))
        
        Whitt1Plus = whitw(-xieff*(1j), 1/2 + s, -2j*r)
        Whitt2Plus = whitw(1-xieff*(1j), 1/2 + s, -2j*r)

        Whitt1Minus = whitw(xieff*(1j), 1/2 + s, -2j*r)
        Whitt2Minus = whitw(1+xieff*(1j), 1/2 + s, -2j*r)
            
        exptermPlus = np.exp(np.pi*xieff)
        exptermMinus = np.exp(-np.pi*xieff)
        
        Fterm = np.zeros((3, 2))

        Fterm[0,0] = exptermPlus*abs((1j*r - 1j*xieff -s) * Whitt1Plus + Whitt2Plus)**2/r**2
        Fterm[0,1] = exptermMinus*abs((1j*r + 1j*xieff -s) * Whitt1Minus + Whitt2Minus)**2/r**2

        Fterm[1,0] = exptermPlus*abs(Whitt1Plus)**2
        Fterm[1,1] = exptermMinus*abs(Whitt1Minus)**2

        Fterm[2,0] = exptermPlus*((Whitt2Plus*Whitt1Plus.conjugate()).real - s * abs(Whitt1Plus)**2)/r
        Fterm[2,1] = exptermMinus*((Whitt2Minus*Whitt1Minus.conjugate()).real - s * abs(Whitt1Minus)**2)/r

        return Fterm

    def WhittakerExactkS(x):
        xi = x.vals["xi"]
        r = x.vals["kS"]/(x.vals["a"]*x.vals["H"])
        
        Whitt1Plus = whitw(-xi*(1j), 1/2, -2j*r)
        Whitt2Plus = whitw(1-xi*(1j), 1/2, -2j*r)

        Whitt1Minus = whitw(xi*(1j), 1/2, -2j*r)
        Whitt2Minus = whitw(1+xi*(1j), 1/2, -2j*r)
            
        exptermPlus = np.exp(np.pi*xi)
        exptermMinus = np.exp(-np.pi*xi)
        
        Fterm = np.zeros((3, 2))

        Fterm[0,0] = exptermPlus*abs(1j*(r - xi) * Whitt1Plus + Whitt2Plus)**2/r**2
        Fterm[0,1] = exptermMinus*abs(1j*(r + xi) * Whitt1Minus + Whitt2Minus)**2/r**2

        Fterm[1,0] = exptermPlus*abs(Whitt1Plus)**2
        Fterm[1,1] = exptermMinus*abs(Whitt1Minus)**2

        Fterm[2,0] = exptermPlus*((Whitt2Plus*Whitt1Plus.conjugate()).real)/r
        Fterm[2,1] = exptermMinus*((Whitt2Minus*Whitt1Minus.conjugate()).real)/r

        return Fterm
    
    def Whittaker_PostFermionEntry(x):
        xieff = x.vals["xieff"]
        xi = x.vals["xi"]
        sB = xieff-xi
        sE = x.vals["s"]
        #z = 2j*x.zferm #-2j*k/aH = 2jkn
        k = x.vals["kh"]
        z = 2j*k*x.etaf(x.vals["t"])
        
        W = np.array([complex(whitw(-1j*xi, 1/2, z)), complex(whitw(1j*xi, 1/2, z))])
        W1 = np.array([complex(whitw(1-1j*xi, 1/2, z)), complex(whitw(1+1j*xi, 1/2, z))])
                      
        Mf = np.array([complex(whitm(-1j*xieff, 1/2+sE, z)), complex(whitm(1j*xieff, 1/2+sE, z))])
        Mf1 = np.array([complex(whitm(1-1j*xieff, 1/2+sE, z)), complex(whitm(1+1j*xieff, 1/2+sE, z))])
                    
        Wf = np.array([complex(whitw(-1j*xieff, 1/2+sE, z)), complex(whitw(1j*xieff, 1/2+sE, z))])
        Wf1 = np.array([complex(whitw(1-1j*xieff, 1/2+sE, z)), complex(whitw(1+1j*xieff, 1/2+sE, z))])
        
        lam = np.array([1., -1.])
        
        Gamma = np.array([complex((gamma(1+sE+1j*lam[i]*xieff)/gamma(2*(1+sE))))/z for i in range(2)])
        
        C = Gamma*(W*(Mf*(sE+1j*lam*sB) + (1+sE-1j*lam*xieff)*Mf1) + W1*Mf)
        D = -Gamma*(W*(Wf*(sE+1j*lam*sB) - Wf1) + W1*Wf)
        
        r = (abs(xieff) + np.sqrt(xieff**2 + sE**2 + sE))
        
        Mr = np.array([complex(whitm(-1j*xieff, 1/2+sE, -2j*r)), complex(whitm(1j*xieff, 1/2+sE, -2j*r))])
        Mr1 = np.array([complex(whitm(1-1j*xieff, 1/2+sE, -2j*r)), complex(whitm(1+1j*xieff, 1/2+sE, -2j*r))])
                    
        Wr = np.array([complex(whitw(-1j*xieff, 1/2+sE, -2j*r)), complex(whitw(1j*xieff, 1/2+sE, -2j*r))])
        Wr1 = np.array([complex(whitw(1-1j*xieff, 1/2+sE, -2j*r)), complex(whitw(1+1j*xieff, 1/2+sE, -2j*r))])
        
        pre = np.exp(np.pi*xi*lam)
        
        Ak = (C*Wr + D*Mr)
        Dk = ((1j*(r-lam*xieff) - sE)*Ak - D*(1+sE-1j*lam*xieff)*Mr1 + C*Wr1)/r
        
        Fterm = np.zeros((3, 2))
        Fterm[0,0] = pre[0]*abs(Dk[0])**2
        Fterm[1,0] = pre[0]*abs(Ak[0])**2
        Fterm[2,0] = pre[0]*(Dk[0]*Ak[0].conjugate()).real
        
        Fterm[0,1] = pre[1]*abs(Dk[1])**2
        Fterm[1,1] = pre[1]*abs(Ak[1])**2
        Fterm[2,1] = pre[1]*(Dk[1]*Ak[1].conjugate()).real

        return Fterm
                  
    """def Whittaker_Interp(x):
        t = x.vals["t"]
        F = np.zeros((3, 2))
        F[0,0] = x.Epf(t)
        F[0,1] = x.Epf(t)
        F[1,0] = x.Bpf(t)
        F[1,1] = x.Bmf(t)
        F[2,0] = x.Gpf(t)
        F[2,1] = x.Gmf(t)
        return F"""

    def EndOfInflation(x, tol=1e-4):
        if x.units == True:
            unitswereon = True
            x.Unitless()
        else:
            unitswereon = False

        N = x.vals["N"]
        dphi = x.vals["dphi"]
        V = x.potential()
        E = x.vals["E"]
        B = x.vals["B"]
        rhoChi = x.vals["rhoChi"]
        f = CubicSpline(N, (dphi**2 - V + (0.5*(E+B) + rhoChi)*x.omega**2/x.f**2))
        res = fsolve(f, max(N), 1e-4)[0]
        
        x.Nend = res

        if unitswereon:
            x.Unitful()
        return res
            
