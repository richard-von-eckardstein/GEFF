import pandas as pd
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.special import binom
from timer import Timer
import math
from mpmath import whitw, re, conj, gamma
import time

class Timer:

    def __init__(self):

        self._start_time = None


    def start(self):

        """Start a new timer"""

        if self._start_time is not None:

            raise TimerError(f"Timer is running. Use .stop() to stop it")


        self._start_time = time.perf_counter()


    def stop(self):

        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:

            raise TimerError(f"Timer is not running. Use .start() to start it")


        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None

        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


class GEF:
    """
    A class used to solve the GEF equations given a set of initial conditions. The class also comes with several useful utility functions.
    
    ...
    
    Attributes
    ----------
    
    completed :  boolean
        Gives information if the GEF equations have been solved or if the GEF is stll to be evaluated
    alpha : int
        Parameter allwoing to switch between physical time and alpha time. Currently, alpha=0 is hardcoded (equations are solved in physical time t)
    beta : float
        Coupling strength of the inflaton to the gauge fields, I_2(phi) = beta7Mpl
    mass : 
        Inflaton mass
    ntr : int
        order in bilinear expectation values at which GEF equations are truncated
    vals : dictionary
        Gives the values of all relevant background quantities. It contains the following keys:
            "t" : physical time
            "N" : e-folds
            "a" : scale-factor
            "H" : Hubble rate
            "phi" : inflaton field value
            "dphi" :  inflaton velocity
            "E" : electric bilinear expectation values (E rot^n E)
            "B" : magnetic bilinear expectation values (B rot^n B)
            "G" : mixed bilinear expectation values (B rot^n B)
            "xi" :  gauge-field production parameter
            "kh" : cut-off scale
        if x.completed = True: the dictionary contains the time-evolution for every parameter
            We only keep "E", "B", "G" for n=0, all higher terms in curls are thrown away to save space
        if x.completed = False:
            if the GEF equations have not been solved, x.vals simply contains the initial condicitons for the above quantities
            if the GEF equations were not solved unsuccessfully, x.vals contains the last point in time which was successfully 
            obtained
    approx : Boolean
        Are Whittaker functions for xi>4 evaluated explicitly or using approximate expressions with rel error < 1e-4
    units : Boolean
        Are all quantities treated as dimensionful or dimensioneless?
    omega :  float
        used for converting from unitless system to unitful quantities and back
    f : float
        used for converting from unitless system to unitful quantities and back
    ratio : float
        omega/f
    H0 : float
        initial value of the Hubble rate in the dimensional unit system. Used for unit conversion.
    Mpl : float
        value of the Planck mass in the dimensional unit system. Used for unit conversion.
        
    ...
    
    Methods
    -------
    
    RunGEF()
        Primary function, execute to solve the GEF equations and store all results in self.vals
    potential() 
        computes the value of the inflaton potential from self.vals
    dVdphi()
        computes the derivative of the inflaton potential from self.vals
    dIdphi()
        computes the derivative of the axial coupling from self.vals, typically dIdphi = beta/Mpl = const.
    ddIddphi()
        computes the 2nd derivative of the axial coupling from self.vals, typically ddIddphi = 0	
    FriedmannEq()
        computes the Friedmann equation from self.vals
    GetXi()
        computes xi from self.vals
    EoMPhi()
        evaluates the Klein--Gordon equation from self.vals
    EoMlnkh()
        computes the time derivative of ln(kh)
    EoMF()
        computes the time derivative of (X rot^n Y) for X,Y in {E, B}
    InitialiseGEF()
        Create an array from self.vals to serve as initial conditions for solve_ivp (only sensible if x.completed=False)
    TimeStep(t, y)
        Evolves the GEF system of equations by one time step, can be passed to solve_ivp
    DefineDictionary(t, y)
        stores the values in t and y from TimeStep() into the dictionary self.vals
    SolveGEF(t0=0., t1=120., atol=1e-6, rtol=1e-3)
        Solve the GEF system from time t0 to t1 with the given tolerances for solve_ivp
    SaveData(outdir)
        store the GEF results in a file under the path specified in outdir
    LoadData(file)
        load GEF results from the specified file
    Unitless()
        Convert all values in self.vals and all EoMs such that they assume dimensionless quantities
    Unitful()
        Convert all values in self.vals and all EoMs such that they assume dimensionful quantities 
    EndOfInflation()
        Find the end of inflation from self.vals by solving for ddot(a) = 0.
    """
 
    def __init__(x, beta, Mpl, ini, M, ntr, approx=True):
        #beta: axial coupling strenght I_2(phi) = beta/M_P
        #Mpl: numerical value of the Planck mass. In Planck units, Mpl = 1
        #ini: a dictionary specifying the initial conditions for the GEF. 
            ##Necessary dictionary keys:
                ### "phi": initial value of inflaton field expressed in the same units as Mpl.
                ### "dphi": initial inflaton field velocity expressed in the same units as Mpl.
        #M: set self.mass
        #ntr: sets self.ntr
        #approx: sets self.approx

        x.units = True
        x.completed = False
        x.alpha = 0
        x.beta = beta
        x.mass = M
        x.vals = ini.copy()
        x.ntr = ntr+1
        if(approx):
            x.Whittaker = x.WhittakerApprox
        else:
            x.Whittaker = x.WhittakerExact
        x.approx = approx
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

    def ddVddphi(x):
        phi = x.f*x.vals["phi"]
        ddV = x.mass**2
        return ddV/(x.omega**2)

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
                       (x.potential() + x.ratio**2*( 0.5*(x.vals["E"][0]+x.vals["B"][0]) ) ) )
        return np.sqrt(Hsq)

    def FriedmannEq2(x):
        Hprime = (-1/2)*(x.vals["dphi"]**2/2 - x.vals["a"]**(2*x.alpha)*
                       (x.potential() + x.ratio**2*(x.vals["E"][0]+x.vals["B"][0])/6) ) - 3/2*x.vals["H"]**2
        return Hprime
    
    def GetXi(x):
        return (x.dIdphi() * x.vals["dphi"])/(2 * x.vals["H"])
            
    #Equations of Motions
    def EoMphi(x):
        alpha = x.alpha
        a = x.vals["a"]
        
        ddphiddt = ((alpha-3)*x.vals["H"] * x.vals["dphi"]
                - a**(2*alpha)*x.dVdphi() - a**(2*alpha)*x.dIdphi()*x.vals["G"][0]*x.ratio**2)
        return ddphiddt
    
    def EoMlnkh(x, ddphiddt):
        kh = x.vals["kh"]
        alpha = x.alpha
        a = x.vals["a"]
        H = x.vals["H"]
        
        xi = x.vals["xi"]
        r = 2*abs(xi)
        
        fc = a**(1-alpha) * H * r
        
        dHdt = 0#x.vals["Hprime"]# #approximation  dHdt = alphaH**2  (slow-roll)
        xiprime = (-dHdt * xi + (x.ddIddphi()*x.vals["dphi"]**2 + x.dIdphi()*ddphiddt)/2)/H
        rprime = 2*np.sign(xi)*xiprime
        fcprime = (1-alpha)*H*fc + dHdt*a**(1-alpha)*r + a**(1-alpha)*H*rprime
                   
        if (fcprime >= 0):
            if((kh-fc)/kh <=1e-3):
                dlnkhdt = fcprime/kh
            else:
                dlnkhdt = 0
        else:
            dlnkhdt = 0
    
        return dlnkhdt

    def EoMF(x, dlnkhdt):
        prefac = dlnkhdt / (4*np.pi**2)

        aAlpha = x.vals["a"]**x.alpha
        H = x.vals["H"]
        E = x.vals["E"]
        B = x.vals["B"]
        G = x.vals["G"]
        kh = x.vals["kh"]
        a = x.vals["a"]
        scale = kh/a

        Whitt = x.Whittaker()

        Whitt[2,1] = -Whitt[2,1]

        bdrF = prefac*np.array([[(scale)**(i+4)*(Whitt[j,0] + (-1)**i*Whitt[j,1]) for j in range(3)]
                                    for i in range(x.ntr)])

        ScalarCpl = (x.dIdphi()*x.vals["dphi"])

        dFdt = np.zeros(bdrF.shape)

        for n in range(x.ntr-1):
            dFdt[n,0] = (bdrF[n, 0] - (4+n)*H*E[n] - 2*aAlpha*G[n+1] + 2*ScalarCpl*G[n])

            dFdt[n,1] = bdrF[n, 1] - (4+n)*H*B[n] + 2*aAlpha*G[n+1]

            dFdt[n,2] = (bdrF[n, 2] - (4+n)*H*G[n] + aAlpha*(E[n+1] - B[n+1]) + ScalarCpl*B[n])

        dFdt[-1,0] = (bdrF[-1,0] -  (4+x.ntr-1)*H*E[-1]- 2*scale**2 * aAlpha*G[-2] + 2*ScalarCpl*G[-1])

        dFdt[-1,1] = bdrF[-1,1] - (4+x.ntr-1)*H*B[-1] + 2*scale**2 * aAlpha*G[-2]

        dFdt[-1,2] = (bdrF[-1,2] - (4+x.ntr-1)*H*G[-1] + scale**2 * aAlpha*(E[-2] - B[-2]) + ScalarCpl*B[-1])

        return dFdt
            
    #Run GEF
    def InitialiseGEF(x):
        yini = np.zeros((x.ntr*3+4))
        
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
                
        dFdt = x.EoMF(dlnkhdt)
        dydt[4:] = dFdt.reshape(x.ntr*3)
        
        return dydt
    
    def DefineDictionary(x, t, y):
        x.vals["t"] = t
        x.vals["N"] = y[0]
        
        x.vals["a"]= np.exp(y[0])
        
        x.vals["phi"] = y[1]
        x.vals["dphi"] = y[2]
        
        F = y[4:]
        F = F.reshape(x.ntr, 3)
        x.vals["E"] = F[:,0]
        x.vals["B"] = F[:,1]
        x.vals["G"] = F[:,2]
        
        x.vals["kh"] = np.exp(y[3])
        x.vals["H"] = x.FriedmannEq()
        x.vals["Hprime"] = x.FriedmannEq2()
        x.vals["xi"] = x.GetXi()
        
        return
    
    def SolveGEF(x, t0=0., t1=120., atol=1e-6, rtol=1e-3):
        t = Timer()
        yini = x.InitialiseGEF()
        ODE = lambda t, y: x.TimeStep(t, y)
        t.start()
        sol = solve_ivp(ODE, [t0,t1], yini, method="RK45", atol=atol, rtol=rtol)
        t.stop()
        return sol
    
    def RunGEF(x, t0=0., t1=120., atol=1e-6, rtol=1e-3):
        if not(x.completed):
            sol = x.SolveGEF(t0, t1, atol=atol, rtol=rtol)
            t = sol.t
            y = sol.y
            print("success:", sol.success)
            parsold = list(x.vals.keys())
            newpars = ["E1", "B1", "G1", "Edot", "Bdot", "Gdot", "ddphi", "dlnkh"]
            pars = parsold + newpars
            res = dict(zip(pars, [[] for par in pars]))
            for i in range(len(t)):
                x.DefineDictionary(t[i], y[:,i])
                ddphi = x.EoMphi()
                res["ddphi"].append(ddphi)
                dlnkhdt = x.EoMlnkh(ddphi)
                res["dlnkh"].append(dlnkhdt)
                dFdt = x.EoMF(dlnkhdt)
                res["Edot"].append(dFdt[0,0])
                res["Bdot"].append(dFdt[1,0])
                res["Gdot"].append(dFdt[2,0])
                for par in parsold:
                    if (par in ["E", "B", "G"]):
                        res[par].append(x.vals[par][0])
                        res[par+"1"].append(x.vals[par][1])
                    else:
                        res[par].append(x.vals[par])
            for par in pars:
                res[par] = np.array(res[par])
            x.vals = res
            x.completed = True
            return sol
        else:
            print("This run is already completed, access data using GEF.vals")
            return
        
    def SaveData(x, outdir):
        if (x.completed):
            #x.Unitful()
            #Data is always stored without units
            x.Unitless()
            path = outdir
            

            output_df = pd.DataFrame(x.vals)  
            output_df.to_csv(path)
        else:
            print("You need to RunGEF first")
            
    def LoadData(x, file):
        input_df = pd.read_table(file, sep=",")
        data = dict(zip(input_df.columns[1:],input_df.values[1:,1:].T))
        
        names = ["t", "phi", "dphi", "ddphi", "a", "H", "E", "B", "G", "E1", "B1", "G1", "Edot", "Bdot", "Gdot", "kh", "dlnkh"]
        #Check if data file is in order:
        for name in names:
            if name not in data.keys():
                print("The file you provided does not contain information on the parameter " + name + ". Please provide a complete data file")
                print("A complete file contains information on the parameters:", names)
                return
            
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
            x.vals["phi"] = x.vals["phi"]/f
            x.vals["dphi"] = x.vals["dphi"]/(f*omega)
            x.vals["ddphi"] = x.vals["ddphi"]/(f*omega**2)
            x.vals["H"] = x.vals["H"]/(omega)
            x.vals["Hprime"] = x.vals["Hprime"]/(omega)**2
            x.vals["E"] = x.vals["E"]/(omega)**4
            x.vals["B"] = x.vals["B"]/(omega)**4
            x.vals["G"] = x.vals["G"]/(omega)**4
            x.vals["kh"] = x.vals["kh"]/omega
            x.vals["E1"] = x.vals["E1"]/(omega)**5
            x.vals["B1"] = x.vals["B1"]/(omega)**5
            x.vals["G1"] = x.vals["G1"]/(omega)**5
            x.vals["Edot"] = x.vals["Edot"]/(omega)**5
            x.vals["Bdot"] = x.vals["Bdot"]/(omega)**5
            x.vals["Gdot"] = x.vals["Gdot"]/(omega)**5
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
            x.vals["ddphi"] = x.vals["ddphi"]*(f*omega**2)
            x.vals["H"] = x.vals["H"]*(omega)
            x.vals["Hprime"] = x.vals["Hprime"]*(omega)**2
            x.vals["E"] = x.vals["E"]*(omega)**4
            x.vals["B"] = x.vals["B"]*(omega)**4
            x.vals["G"] = x.vals["G"]*(omega)**4
            x.vals["kh"] = x.vals["kh"]*omega
            x.vals["E1"] = x.vals["E1"]*(omega)**5
            x.vals["B1"] = x.vals["B1"]*(omega)**5
            x.vals["G1"] = x.vals["G1"]*(omega)**5
            x.vals["Edot"] = x.vals["Edot"]*(omega)**5
            x.vals["Bdot"] = x.vals["Bdot"]*(omega)**5
            x.vals["Gdot"] = x.vals["Gdot"]*(omega)**5
            x.omega = 1.
            x.f = 1.
            x.units = True
        else:
            print("Already Unitful")
        return
                
    #Whittaker Functions
    def WhittakerApprox(x):
        xi = x.vals["xi"]
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
    
    def WhittakerExact(x):
        xieff = x.vals["xi"]
        s = 0.
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
    
    def EndOfInflation(x, tol=1e-4, plot=False):
        if x.units == True:
            unitswereon = True
            x.Unitless()
        else:
            unitswereon = False
        print(x.omega)
        N = x.vals["N"]
        dphi = x.vals["dphi"]
        V = x.potential()
        E = x.vals["E"]
        B = x.vals["B"]
        f = CubicSpline(N, (dphi**2 - V + 0.5*(E+B)*x.omega**2/x.f**2))
        res = fsolve(f, 60, 1e-4)
        print(res)
        if unitswereon:
            x.Unitful()
        if plot:
            plt.plot(N, f(N))
            plt.plot(N, np.zeros(N.size))
            plt.xlim(60, max(N))
            plt.ylim(-0.01,0.01)
            plt.vlines(res, -1e-2, 1e-2, "k")
            plt.show()       
        return res
   
            
