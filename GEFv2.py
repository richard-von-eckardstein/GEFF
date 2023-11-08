import pandas as pd
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from timer import Timer
import math
from mpmath import whitw, whitm, re, conj

class GEF:
    def __init__(x, alpha, beta, Mpl, ini, M, ntr, SE, AltDamp=0, approx=True):
        x.units = True
        x.completed = False
        x.alpha = alpha
        x.beta = beta
        x.mass = M
        x.SE = SE
        x.vals = ini.copy()
        x.AltDamp = AltDamp
        if (SE==None):
            x.GaugePos = 4
            x.conductivity = lambda y: 0., 0.
            x.vals["rhoChi"] = 0.
            x.vals["delta"] = 1.
            x.vals["sigmaE"] = 0.
            x.vals["s"] = 0.
            x.vals["sigmaB"] = 0.
            if(approx):
                x.Whittaker = x.WhittakerApprox_NoSE
            else:
                x.Whittaker = x.WhittakerExact
            #x.ODE = x.fullGEF_NoSE()
        else:
            if(approx):
                x.Whittaker = x.WhittakerApprox_WithSE
            else:
                x.Whittaker = x.WhittakerExact
                
            if (SE=="mix"):
                if (x.AltDamp == 1):
                    x.GaugePos = 5
                    x.conductivity = x.ComputeImprovedSigmaAltDelta
                elif (x.AltDamp == 1):
                    x.GaugePos = 5
                    x.conductivity = x.ComputeImprovedSigmaAltDelta
                else:
                    x.GaugePos = 6
                    x.conductivity = x.ComputeImprovedSigma
            elif (-1. <= SE <=1.):
                if (x.AltDamp == 1):
                    x.GaugePos = 5
                    x.conductivity = x.ComputeSigmaCollinearAltDelta
                elif (x.AltDamp == 2):
                    x.DeltaFunc = False
                    x.GaugePos = 7
                    #x.GaugePos = 6
                    x.conductivity = x.ComputeSigmaCollinearKDep
                else:
                    x.GaugePos = 6
                    x.conductivity = x.ComputeSigmaCollinear
                if(SE == 1. and approx):
                    #Only xieff, no s in pure magnetic picture
                    x.Whittaker = x.WhittakerApprox_NoSE
            else:
                print(SE, "is not a valid choice for SE")
        x.approx = approx
        x.ntr = ntr+1
        x.omega = 1.
        x.f = 1.
        x.ratio = 1.
        #Need Unitful Potential once, to compute omega
        x.H0 = np.sqrt((0.5*x.vals["dphi"]**2 + x.potential())/(3*Mpl**2))
        x.Mpl = Mpl
    
    #Potentials and Couplings
    def potential(x):
        phi = x.f*x.vals["phi"]
        V = 0.5*phi**2*x.mass**2
        return V / (x.f*x.omega)**2
    
    def dVdphi(x):
        phi = x.f*x.vals["phi"]
        dV = phi*x.mass**2
        return dV/(x.f*x.omega**2)

    def dIdphi(x):
        phi = x.f*x.vals["phi"]
        dI = x.beta/x.f
        return dI*x.f

    def ddIddphi(x):
        phi = x.f*x.vals["phi"]
        ddI = 0.
        return ddI*x.f**2
    
    #Useful Quantities
    def FriedmannEq(x):
        Hsq = (1/3) * (0.5 * x.vals["dphi"]**2 + x.vals["a"]**(2*x.alpha)*
                       (x.potential() + x.ratio**2*(0.5*(x.vals["E"][0]+x.vals["B"][0]) + x.vals["rhoChi"])))
        return np.sqrt(Hsq)
    
    def GetXi(x):
        return (x.dIdphi() * x.vals["dphi"])/(2 * x.vals["H"])
    
    def GetS(x, sigma):
        return (x.vals["a"]**(x.alpha) * sigma)/(2* x.vals["H"])
    
    def ComputeSigmaCollinear(x):
        E0 = x.vals["E"][0]
        B0 = x.vals["B"][0]
        mu = ((E0+B0)/2)**(1/4)
        if mu==0.:
            return 0., 0.
        else:
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))

            frac = x.SE
            sigma = ((x.vals["a"]**x.alpha) * (41.*gmu**3/(72.*np.pi**2 * x.vals["H"] * np.tanh(np.pi*np.sqrt(B0/E0)))))
            sigmaE =  np.sqrt(B0) * (min(1., 1.- frac)*E0 + max(-frac, 0.)*B0) * sigma / (E0+B0)         
            sigmaB = -np.sign(x.vals["G"][0]) * np.sqrt(E0)*(min(1., 1.+ frac)*B0 + max(frac,0.)*E0)* sigma/(E0+B0)
            
            return sigmaE, sigmaB
        
    def ComputeSigmaCollinearAltDelta(x):
        E0 = x.vals["E"][0]
        B0 = x.vals["B"][0]
        mu = ((E0+B0)/2)**(1/4)
        if mu==0:
            return 0., 0., 1., 0.
        else:
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))
            
            H = x.vals["H"]

            frac = x.SE
            sigma = ((x.vals["a"]**x.alpha) * (41.*gmu**3/(72.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B0/E0)))))
            sigmaE =  np.sqrt(B0) * (min(1., 1.- frac)*E0 + max(-frac, 0.)*B0) * sigma / (E0+B0)         
            sigmaB = -np.sign(x.vals["G"][0]) * np.sqrt(E0)*(min(1., 1.+ frac)*B0 + max(frac,0.)*E0)* sigma/(E0+B0)
            
            kh = x.vals["kh"]*x.vals["a"]**(x.alpha-1)
            
            kappa = np.log(gmu**(1/2)*E0**(1/4)/kh)
            
            if (kappa > 0.):
                return sigmaE*0.27, sigmaB, np.exp(-sigmaE/H*kappa), x.GetS(sigmaE*0.27)
            else:
                return 0., sigmaB, 1., 0.
            
    def ComputeSigmaCollinearKDep(x):
        E0 = x.vals["E"][0]
        B0 = x.vals["B"][0]
        mu = (E0+B0)/2
        #print(mu)
        if mu==0:
            return 0., 0., 1e-2*x.vals["kh"]
        else:
            mu = (mu)**(1/4)
            mz = 91.2/(2.43536e18)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))
            
            H = x.vals["H"]

            frac = x.SE
            sigma = ((x.vals["a"]**x.alpha) * (41.*gmu**3/(72.*np.pi**2 * H * np.tanh(np.pi*np.sqrt(B0/E0)))))
            sigmaE =  np.sqrt(B0) * (min(1., 1.- frac)*E0 + max(-frac, 0.)*B0) * sigma / (E0+B0)         
            sigmaB = -np.sign(x.vals["G"][0]) * np.sqrt(E0)*(min(1., 1.+ frac)*B0 + max(frac,0.)*E0)* sigma/(E0+B0)
            
            ks = gmu**(1/2)*E0**(1/4)*x.vals["a"]**(1-x.alpha)
            #print(gmu, E0)
            return sigmaE, sigmaB, ks
            
            
    def ComputeImprovedSigma(x):
        E0 = x.vals["E"][0]
        B0 = x.vals["B"][0]
        G0 = x.vals["G"][0]
        Sigma = np.sqrt((E0 - B0)**2 + 4*G0**2)
        #print(Sigma)
        if Sigma==0:
            x.vals["sigmaE"] = 0.
            x.vals["sigmaB"] = 0.
            return 0., 0.
        else:
            mz = 91.2/(2.43536e18)
            mu = ((Sigma)/2)**(1/4)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))

            Eprime = np.sqrt(E0 - B0 + Sigma)
            Bprime = np.sqrt(B0- E0 + Sigma)
            Sum = E0 + B0 + Sigma

            sigma = (x.vals["a"]**x.alpha)*(41.*gmu**3/(72.*np.pi**2)
                     /(np.sqrt(Sigma*Sum)*x.vals["H"] * np.tanh(np.pi*Bprime/Eprime)))
            
            return abs(G0)*Eprime*sigma, -G0*Bprime*sigma
        
        
    def ComputeImprovedSigmaAltDelta(x):
        E0 = x.vals["E"][0]
        B0 = x.vals["B"][0]
        G0 = x.vals["G"][0]
        Sigma = np.sqrt((E0 - B0)**2 + 4*G0**2)
        #print(Sigma)
        if Sigma==0:
            x.vals["sigmaE"] = 0.
            x.vals["sigmaB"] = 0.
            return 0., 0., 1., 0.
        else:
            mz = 91.2/(2.43536e18)
            mu = ((Sigma)/2)**(1/4)
            gmz = 0.35
            gmu = np.sqrt(gmz**2/(1 + gmz**2*41./(48.*np.pi**2)*np.log(mz/(mu*x.ratio))))

            Eprime = np.sqrt(E0 - B0 + Sigma)
            Bprime = np.sqrt(B0- E0 + Sigma)
            Sum = E0 + B0 + Sigma
            
            H = x.vals["H"]
            
            sigma = (x.vals["a"]**x.alpha)*(41.*gmu**3/(72.*np.pi**2)
                     /(np.sqrt(Sigma*Sum)* H * np.tanh(np.pi*Bprime/Eprime)))
            
            sigmaE = abs(G0)*Eprime*sigma
            sigmaB = -G0*Bprime*sigma
            
            #s = x.GetS(sigmaE)
            #xieff = x.vals["xi"] + x.GetS(sigmaB)
            #r = abs(xieff) + np.sqrt(xieff**2 + s**2 + s)
            kh = x.vals["kh"]*x.vals["a"]**(x.alpha-1)
            
            kappa = np.log( (gmu*Sigma/Sum)**(1/2) * (2*Eprime)**(1/4)/kh)
            
            if (kappa > 0.):
                return sigmaE, sigmaB, np.exp(-sigmaE/H*kappa), x.GetS(sigmaE)*0.15
            else:
                return sigmaE, sigmaB, 1., 0.
            
            
    #Equations of Motions
    def EoMphi(x):
        alpha = x.alpha
        a = x.vals["a"]
        
        ddphiddt = ((alpha-3)*x.vals["H"] * x.vals["dphi"]
                - a**(2*alpha)*x.dVdphi() - a**(2*alpha)*x.dIdphi()*x.vals["G"][0]*x.ratio**2)
        return ddphiddt
    
    def EoMlnkh(x, ddphiddt):
        alpha = x.alpha
        a = x.vals["a"]
        H = x.vals["H"]
        sigmaE = x.vals["sigmaE"]
        sigmaB = x.vals["sigmaB"]
        kh = x.vals["kh"]
        
        xieff = x.vals["xieff"]
        s = x.vals["s"]
        r = (abs(xieff) + np.sqrt(xieff**2 + s**2 + s))
        
        fc = a**(1-alpha) * H * r

        dsigmaEdt = 0.
        dsigmaBdt = 0.
        #approximation: dHdt = alphaH**2 (slow-roll)
        fcprime = (((1-alpha)*H+a**(1-alpha)*alpha*H)*fc
                      + np.sign(xieff)*(a**(1-alpha)*(x.ddIddphi()*x.vals["dphi"]**2
                                                      + x.dIdphi()*ddphiddt)/2
                                           +a*(alpha*H*sigmaB + dsigmaBdt)/2
                                           - a**(1-alpha)*alpha*H**2*xieff)*
                   (1. + xieff/np.sqrt(xieff**2 + s**2 + s))
                      + (a*(alpha*H*sigmaE + dsigmaEdt)/2 
                         - a**(1-alpha)*alpha*H**2*s)*(abs(s) + 1/2)/np.sqrt(xieff**2 + s**2 + s))
        #fprime = f*H
        if (fcprime >= 0):
            if((kh-fc)/kh <=1e-3):
                dlnkhdt = fcprime/kh
            else:
                dlnkhdt = 0
        else:
            dlnkhdt = 0
    
        return dlnkhdt
    
    
    def CheckFermionEntry(x, dlnkhdtO, dlnkhdtE):
        kh = x.vals["kh"]
        khO = x.vals["khO"]
        khE = x.vals["khE"]
        kS = x.vals["kS"]
        t = x.vals["t"]
        #print(kS)
        kharr = np.array([khO, kh, khE])
        dlnkharr = np.array([dlnkhdtO, 0., dlnkhdtE])
        
        ind1 = int((1+np.sign(np.log(kS/kh)))/2)
        ind2 = int(1+(1+np.sign(np.log(khE/khO)))/2)*ind1
        
        sigmaE = x.vals["sigmaE"]*x.FermionEntry
        sigmaB = x.vals["sigmaB"]*x.FermionEntry
        s = x.vals["s"]*x.FermionEntry
        xieff = x.vals["xieff"]*x.FermionEntry + x.vals["xi"]*(1-x.FermionEntry)
        
        kh = kharr[ind2]
        dlnkh = dlnkharr[ind2]
        
        if((ind1 > 0.) and x.FermionEntry==1):
            if (1e-5 < (t - x.tdep[-1][0])):
                    x.tdep.append([t, kS, sigmaE, x.vals["a"]*x.vals["H"]])
            #x.Whittaker=x.Whittaker_PostFermionEntry
            tf = x.vals["tferm"]
            arr = np.array(x.tdep)
            
            ts = np.array([*list(arr[:,0]), *list(np.linspace(arr[-1,0]*(1.02), 2*arr[-1,0],100))])
            #print(ts)
            kS = np.array([*list(arr[:,1]), *list(1e3*arr[-1,1]*np.ones(100))])
                    
            kSf = CubicSpline(ts, kS)
                    
            f = lambda x: np.log(kSf(x)/kh)
            tferm = fsolve(f, tf, xtol=1e-7)[0]
            #print(x.vals["t"], tferm, f(t), f(tferm))
            ts = arr[:,0]
            kS = arr[:,1]
            sE = arr[:,2]
            aH = arr[:,3]
            sigmatferm = CubicSpline(ts, sE)(tferm)
            dkStferm = CubicSpline((ts[1:]+ts[:-1])/2, (kS[1:]-kS[:-1])/(ts[1:]-ts[:-1]))(tferm)
            aHtferm = CubicSpline(ts, aH)(tferm)
                #etatferm = CubicSpline(ts, eta)(tferm)
                #print("aH", -1/aHtferm, "etatferm", etatferm)
            ddelta = (-sigmaE + sigmatferm*kh*dlnkh/abs(dkStferm))*x.vals["delta"]
            x.zferm = -kh/aHtferm
            x.vals["tferm"] = tferm
        elif ((ind1 > 0.) and x.FermionEntry==0):
            x.FermionEntry=1
            t = x.vals["t"]
            x.vals["tferm"] = t
            if (1e-5 < (t - x.tdep[-1][0])):
                    x.tdep.append([t, kS, sigmaE, x.vals["a"]*x.vals["H"]])
            ddelta = 0.
        else:
            if (1e-5 < (t - x.tdep[-1][0])):
                    x.tdep.append([t, 1e100, sigmaE, x.vals["a"]*x.vals["H"]])
            ddelta=0.
            
        return dlnkh, kh, sigmaE, sigmaB, s, xieff, ddelta
    
    def EoMDelta(x):
        return -x.vals["a"]**(x.alpha)*x.vals["sigmaE"]*x.vals["delta"]
                
    def EoMrhoChi(x):
        return (x.vals["a"]**(x.alpha)*(x.vals["sigmaE"]*x.vals["E"][0]
                                        - x.vals["sigmaB"]*x.vals["G"][0])- 4*x.vals["H"]*x.vals["rhoChi"])
    
    def EoMF(x, dlnkhdt):
        prefac = dlnkhdt * x.vals["delta"] / (4*np.pi**2)
        
        aAlpha = x.vals["a"]**x.alpha
        H = x.vals["H"]
        E = x.vals["E"]
        B = x.vals["B"]
        G = x.vals["G"]
        sigmaE = x.vals["sigmaE"]
        scale = x.vals["kh"]/x.vals["a"]
        
        ScalarCpl = (x.dIdphi()*x.vals["dphi"]+aAlpha*x.vals["sigmaB"])
        
        Whitt = x.Whittaker()
        
        Whitt[2,1] = -Whitt[2,1]
        
        bdrF = prefac*np.array([[(scale)**(i+4)*(Whitt[j,0] + (-1)**i*Whitt[j,1]) for j in range(3)]
                                for i in range(x.ntr)])
        
        dFdt = np.zeros(bdrF.shape)
    
        for n in range(x.ntr-1):
            dFdt[n,0] = (bdrF[n, 0] - ((4+n)*H + 2*aAlpha * sigmaE)*E[n]
                         - 2*aAlpha*G[n+1] + 2*ScalarCpl*G[n])

            dFdt[n,1] = bdrF[n, 1] - ((4+n)*H)*B[n] + 2*aAlpha*G[n+1]

            dFdt[n,2] = (bdrF[n, 2] - ((4+n)*H + aAlpha * sigmaE)*G[n]
                         + aAlpha*(E[n+1] - B[n+1]) + ScalarCpl*B[n])

        dFdt[-1,0] = (bdrF[-1,0] -  ((4+x.ntr-1)*H + 2*aAlpha * sigmaE)*E[-1]
                        - 2*scale**2 * aAlpha*G[-2] + 2*ScalarCpl*G[-1])
        
        dFdt[-1,1] = bdrF[-1,1] - (4+x.ntr-1)*H*B[-1] + 2*scale**2 * aAlpha*G[-2]
        
        dFdt[-1,2] = (bdrF[-1,2] - ((4+x.ntr-1)*H + aAlpha * sigmaE)*G[-1] 
                         + scale**2 * aAlpha*(E[-2] - B[-2]) + ScalarCpl*B[-1])

        return dFdt
    
    #Whittaker Functions
    def WhittakerApprox_NoSE(x):
        xi = x.vals["xieff"]
        if (abs(xi) >= 3):
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
        Fterm[2,1]= exptermMinus*((Whitt2Minus*Whitt1Minus.conjugate()).real - s * abs(Whitt1Minus)**2)/r

        return Fterm
    
    def Whittaker_PostFermionEntry(x):
        xieff = x.vals["xieff"]
        xi = x.vals["xi"]
        sB = xieff-xi
        sE = x.vals["s"]
        z = 2j*x.zferm
        k = x.vals["kh"]
        
        W = np.array([complex(whitw(-1j*xi, 1/2, z)), complex(whitw(1j*xi, 1/2, z))])
        W1 = np.array([complex(whitw(1-1j*xi, 1/2, z)), complex(whitw(1+1j*xi, 1/2, z))])
                      
        Mf = np.array([complex(whitm(-1j*xieff, 1/2+sE, z)), complex(whitm(1j*xieff, 1/2+sE, z))])
        Mf1 = np.array([complex(whitm(1-1j*xieff, 1/2+sE, z)), complex(whitm(1+1j*xieff, 1/2+sE, z))])
                    
        Wf = np.array([complex(whitw(-1j*xieff, 1/2+sE, z)), complex(whitw(1j*xieff, 1/2+sE, z))])
        Wf1 = np.array([complex(whitw(1-1j*xieff, 1/2+sE, z)), complex(whitw(1+1j*xieff, 1/2+sE, z))])
        
        lam = np.array([1., -1.])
        
        C = np.exp(np.pi*xi*lam)/(2*k)
        B = (W*(Wf1/Wf-1j*lam*sB-sE)-W1)/((1.+sE-1j*lam*xieff)*Mf1+Wf1/Wf*Mf)
        A = (W - B*Mf)/Wf
        
        #print("A:", A, "B:", B)
        
        r = (abs(xieff) + np.sqrt(xieff**2 + sE**2 + sE))
        
        Mr = np.array([complex(whitm(-1j*xieff, 1/2+sE, -2j*r)), complex(whitm(1j*xieff, 1/2+sE, -2j*r))])
        Mr1 = np.array([complex(whitm(1-1j*xieff, 1/2+sE, -2j*r)), complex(whitm(1+1j*xieff, 1/2+sE, -2j*r))])
                    
        Wr = np.array([complex(whitw(-1j*xieff, 1/2+sE, -2j*r)), complex(whitw(1j*xieff, 1/2+sE, -2j*r))])
        Wr1 = np.array([complex(whitw(1-1j*xieff, 1/2+sE, -2j*r)), complex(whitw(1+1j*xieff, 1/2+sE, -2j*r))])
        
        Akplus = A[0]*Wr[0] + B[0]*Mr[0]
        Akminus = A[1]*Wr[1] + B[1]*Mr[1]
        Dkplus = ((1j*(r-xieff) - sE)*Akplus + A[0]*Wr1[0] - B[0]*(1+sE-1j*xieff)*Mr1[0])/r
        Dkminus = ((1j*(r+xieff) - sE)*Akminus + A[1]*Wr1[1] - B[1]*(1+sE+1j*xieff)*Mr1[1])/r
        
        Fterm = np.zeros((3, 2))
        Fterm[0,0] = C[0]*abs(Dkplus)**2
        Fterm[1,0] = C[0]*abs(Akplus)**2
        Fterm[2,0] = C[0]*(Dkplus*Akplus.conjugate()).real
        
        Fterm[0,1] = C[1]*abs(Dkminus)**2
        Fterm[1,1] = C[1]*abs(Akminus)**2
        Fterm[2,1] = C[1]*(Dkminus*Akminus.conjugate()).real
        
        return Fterm
    
    #Run GEF
    def InitialiseGEF(x):
        yini = np.zeros((x.ntr*3+x.GaugePos))

        if (x.units):
            x.f = x.Mpl
            x.omega = x.H0
            x.units = False
        else:
            x.f = 1.
            x.omega = 1.
            x.units = True
        yini[0] = 0
        yini[1] = x.vals["phi"]/x.f
        yini[2] = x.vals["dphi"]/(x.f*x.omega)
        x.vals["kh"] = np.log(abs(yini[2]*x.dIdphi()))
        yini[3] = x.vals["kh"]
        
        if (x.SE != None):
            if (x.AltDamp == 1):
                yini[4] = x.vals["rhoChi"]
            elif (x.AltDamp == 2):
                x.FermionEntry = 0
                yini[4] = yini[3]
                yini[5] = x.vals["delta"]
                yini[6] = x.vals["rhoChi"]
                #yini[5] = x.vals["rhoChi"]
                x.tdep = [[x.vals["t"], 0., 0., 1.]]
            else:
                yini[4] = x.vals["delta"]
                yini[5] = x.vals["rhoChi"]
        #print(yini)
        x.f = x.Mpl
        x.omega = x.H0
        x.ratio = x.omega/x.f
        x.units = False
        
        return yini
    
    def TimeStep(x, t, y):
        x.DefineDictionary(t, y)
        dydt = np.zeros(y.shape)
        dydt[0] = x.vals["H"]
        dydt[1] = x.vals["dphi"]
        dydt[2] = x.EoMphi()
        dlnkhdt = x.EoMlnkh(dydt[2])
        dydt[3] = dlnkhdt

        if (x.SE != None):
            if (x.AltDamp == 1):
                dlnkhdt = x.EoMlnkh(dydt[2])
                dydt[3] = dlnkhdt
                dydt[4] = x.EoMrhoChi()
            elif (x.AltDamp == 2):
                sigmaEtmp = x.vals["sigmaE"]
                sigmaBtmp = x.vals["sigmaB"]
                x.vals["sigmaE"] = 0.
                x.vals["sigmaB"] = 0.
                dlnkhdt = x.EoMlnkh(dydt[2])
                dydt[3] = dlnkhdt
                x.vals["sigmaE"] = sigmaEtmp
                x.vals["sigmaB"] = sigmaBtmp 
                x.vals["s"] = x.GetS(x.vals["sigmaE"])
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])
                dydt[4] = x.EoMlnkh(dydt[2])
                dlnkhdt, x.vals["kh"], x.vals["sigmaE"], x.vals["sigmaB"], x.vals["s"], x.vals["xieff"], ddelta = x.CheckFermionEntry(dydt[3], dydt[4])
                #x.vals["delta"] = np.exp(-x.vals["sigmaE"]/x.vals["H"]*np.log(x.vals["kS"]/x.vals["kh"]))
                dydt[5] = ddelta
                dydt[6] = x.EoMrhoChi()
                #dydt[6] = x.EoMrhoChi()
            else:
                dlnkhdt = x.EoMlnkh(dydt[2])
                dydt[3] = dlnkhdt
                dydt[4] = x.EoMDelta()
                dydt[5] = x.EoMrhoChi()
        else:
            dlnkhdt = x.EoMlnkh(dydt[2])
            dydt[3] = dlnkhdt
            
                
        dFdt = x.EoMF(dlnkhdt)
        dydt[x.GaugePos:] = dFdt.reshape(x.ntr*3)
        
        return dydt
    
    def SolveGEF(x, t0=0., t1=120.):
        t = Timer()
        x.vals["t"] = t0
        yini = x.InitialiseGEF()
        ODE = lambda t, y: x.TimeStep(t, y)
        t.start()
        sol = solve_ivp(ODE, [t0,t1], yini, method="RK45")
        t.stop()
        return sol
    
    def RunGEF(x, t0=0., t1=120.):
        if not(x.completed):
            sol = x.SolveGEF(t0, t1)
            t = sol.t
            y = sol.y
            pars = x.vals.keys()
            res = dict(zip(pars, [[] for par in pars]))
            if (x.AltDamp == 2):
                x.FermionEntry = 0
                res["sigmaEkh"] = []
                res["sigmaBkh"] = []
                res["skh"] = []
                res["xieffkh"] = []
            for i in range(len(t)):
                x.DefineDictionary(t[i], y[:,i])
                if (x.AltDamp == 2):
                    _, x.vals["kh"], x.vals["sigmaEkh"], x.vals["sigmaBkh"], x.vals["skh"], x.vals["xieffkh"], _ = x.CheckFermionEntry(0., 0.)
                for par in pars:
                    if (par in ["E", "B", "G"]):
                        res[par].append(x.vals[par][0])
                    else:
                        res[par].append(x.vals[par])
            for par in pars:
                res[par] = np.array(res[par])
            x.vals = res
            x.completed = True
        else:
            print("This run is already completed, access data using GEF.vals")
        return
    
    def SaveData(x):
        if (x.completed):
            #x.Unitful()
            if(x.AltDamp == 1):
                filename = "Out/GEF_Beta"+str(x.beta)+"_SE"+str(x.SE)+"_AltDamp_v4.dat"
            elif(x.AltDamp == 2):
                filename = "Out/GEF_Beta"+str(x.beta)+"_SE"+str(x.SE)+"_KDep_v3.dat"
            else:
                filename = "Out/GEF_Beta"+str(x.beta)+"_SE"+str(x.SE)+".dat"
            DirName = os.getcwd()
            
            """settings = [{"alpha":x.alpha}, {"beta":x.beta}, {"M":x.mass},{ "Mpl":x.Mpl}, 
                        {"H0":x.H0}, {"units":x.units}, {"SE":x.SE}, {"approx":x.approx}, {"ntr":x.ntr}]"""
            
            path = os.path.join(DirName, filename)
            
            #dic = dict(x.vals, **{"settings":settings})

            output_df = pd.DataFrame(x.vals)  
            output_df.to_csv(path)
        else:
            print("You need to RunGEF first")

    def Unitless(x):
        omega = x.H0
        f = x.Mpl
        if (not(x.completed)):
            print("You need to RunGEF or LoadGEF first")
            return
        if (x.units):
            x.vals["t"] = x.vals["t"]*omega
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
            if (x.AltDamp == 2):
                x.vals["khE"] = x.vals["khE"]/omega
                x.vals["khO"] = x.vals["khO"]/omega
                x.vals["kS"] = x.vals["kS"]/omega
                x.vals["sigmaEkh"] = x.vals["sigmaEkh"]/omega
                x.vals["sigmaBkh"] = x.vals["sigmaBkh"]/omega
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
            if (x.AltDamp == 2):
                x.vals["khE"] = x.vals["khE"]*omega
                x.vals["khO"] = x.vals["khO"]*omega
                x.vals["kS"] = x.vals["kS"]*omega
                x.vals["sigmaEkh"] = x.vals["sigmaEkh"]*omega
                x.vals["sigmaBkh"] = x.vals["sigmaBkh"]*omega
            x.omega = 1.
            x.f = 1.
            x.units = True
        else:
            print("Already Unitful")
        return
      
    def DefineDictionary(x, t, y):
        x.vals["t"] = t
        x.vals["N"] = y[0]
        
        x.vals["a"]= np.exp(y[0])
        
        x.vals["phi"] = y[1]
        x.vals["dphi"] = y[2]
        
        F = y[x.GaugePos:]
        F = F.reshape(x.ntr, 3)
        x.vals["E"] = F[:,0]
        x.vals["B"] = F[:,1]
        x.vals["G"] = F[:,2]
        
        if (x.SE == None):
            x.vals["kh"] = np.exp(y[3])
            x.vals["H"] = x.FriedmannEq()
            x.vals["xi"] = x.GetXi()
            x.vals["xieff"] = x.vals["xi"]
        else:
            if (x.AltDamp == 1):
                x.vals["kh"] = np.exp(y[3])
                x.vals["rhoChi"] = y[4]
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"], x.vals["delta"], x.vals["s"] = x.conductivity()
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])
            elif (x.AltDamp == 2):
                x.vals["khO"] = np.exp(y[3])
                x.vals["khE"] = np.exp(y[4])
                x.vals["delta"] = y[5]
                #print(y[5])
                x.vals["rhoChi"] = y[6]
                #x.vals["rhoChi"] = y[5]
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"], x.vals["kS"] = x.conductivity()
                x.vals["s"] = 0.
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"]
            else:
                x.vals["kh"] = np.exp(y[3])
                x.vals["delta"] = y[4]
                x.vals["rhoChi"] = y[5]
                x.vals["H"] = x.FriedmannEq()
                x.vals["sigmaE"], x.vals["sigmaB"] = x.conductivity()
                x.vals["s"] = x.GetS(x.vals["sigmaE"])
                x.vals["xi"] = x.GetXi()
                x.vals["xieff"] = x.vals["xi"] + x.GetS(x.vals["sigmaB"])
            
