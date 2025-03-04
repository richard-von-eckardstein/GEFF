import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from src.Tools.timer import Timer
import math
from mpmath import whitw, mp

class TruncationError(Exception):
    pass
class UnknownEventError(Exception):
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
        Coupling strength of the inflaton to the gauge fields, I_2(phi) = beta/Mpl
    V : function
        The potential as a function of the inflaton field in Planck units.
    dV : function
        The potential derivative as a function of the inflaton field in Planck units.
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
    GEFData : None | str
        Path to file where GEF results are stored
    ModeData : None | str
        Path to file where Mode By Mode results are stored
        
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
    SolveGEF(ntr, t0=0., t1=120., atol=1e-6, rtol=1e-3)
        Solve the GEF system from time t0 to t1 with the given tolerances for solve_ivp
    SaveData()
        store the GEF results in a file under the path GEF.GEFData or a default path.
    LoadData()
        load GEF results from GEF.GEFData
    Unitless()
        Convert all values in self.vals and all EoMs such that they assume dimensionless quantities
    Unitful()
        Convert all values in self.vals and all EoMs such that they assume dimensionful quantities 
    EndOfInflation()
        Find the end of inflation from self.vals by solving for ddot(a) = 0.
    """
 
    def __init__(x, beta: float, ini: dict, V, dV, GEFData: None|str=None, ModeData: None|str=None, approx: bool=True):
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
        
        x.V = V
        x.dV = dV

        x.ini = ini

        x.GEFData = GEFData
        x.ModeData = ModeData

        if(approx):
            x.Whittaker = x.WhittakerApprox
        else:
            x.Whittaker = x.WhittakerExact
        
        x.approx = approx
        x.omega = 1.
        x.f = 1.
        x.ratio = 1.
        #Need Unitful Potential once, to compute omega
        x.H0 = np.sqrt( ( 0.5*x.ini["dphi"]**2 + x.V(x.ini["phi"]) )/3 )
        x.Mpl = 1.
        x.Nend = 61
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
                       (x.potential() + x.ratio**2*( 0.5*(x.vals["E"]+x.vals["B"]) ) ) )
        return np.sqrt(Hsq)

    def FriedmannEq2(x):
        Hprime = -(x.vals["dphi"]**2/3 - x.vals["a"]**(2*x.alpha)*
                       (x.potential()/3 - x.ratio**2*(x.vals["E"]+x.vals["B"])/6) ) - (1-x.alpha)*x.vals["H"]**2
        return Hprime
    
    def GetXi(x):
        return (x.dIdphi() * x.vals["dphi"])/(2 * x.vals["H"])
            
    #Equations of Motions
    def EoMphi(x):
        alpha = x.alpha
        a = x.vals["a"]
        
        ddphiddt = ((alpha-3)*x.vals["H"] * x.vals["dphi"]
                - a**(2*alpha)*x.dVdphi() - a**(2*alpha)*x.dIdphi()*x.vals["G"]*x.ratio**2)
        return ddphiddt
    
    def EoMlnkh(x, ddphiddt, rtol=1e-6):
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
            if(1-np.log(fc)/np.log(kh) < rtol):
                dlnkhdt = fcprime/kh
            else:
                dlnkhdt = 0
        else:
            dlnkhdt = 0
    
        return dlnkhdt

    def EoMF(x, dlnkhdt):
        FE = x.vals["F"][:,0]
        FB = x.vals["F"][:,1]
        FG = x.vals["F"][:,2]
        aAlpha = x.vals["a"]**x.alpha

        kh = x.vals["kh"]
        a = x.vals["a"]
        scale = kh/a

        Whitt = x.Whittaker()

        Whitt[2,1] = -Whitt[2,1]

        bdrF = dlnkhdt*np.array([[(Whitt[j,0] + (-1)**i*Whitt[j,1]) for j in range(3)]
                                    for i in range(x.ntr)]) / (4*np.pi**2)

        ScalarCpl = (x.dIdphi()*x.vals["dphi"])

        dFdt = np.zeros(bdrF.shape)

        for n in range(x.ntr-1):
            dFdt[n,0] = (bdrF[n, 0] - (4+n)*dlnkhdt*FE[n] - 2*aAlpha*scale*FG[n+1] + 2*ScalarCpl*FG[n])

            dFdt[n,1] = (bdrF[n, 1] - (4+n)*dlnkhdt*FB[n] + 2*aAlpha*scale*FG[n+1])

            dFdt[n,2] = (bdrF[n, 2] - (4+n)*dlnkhdt*FG[n] + aAlpha*scale*(FE[n+1] - FB[n+1]) + ScalarCpl*FB[n])

        dFdt[-1,0] = (bdrF[-1,0] -  (4+x.ntr-1)*dlnkhdt*FE[-1]- 2*aAlpha*scale*FG[-2] + 2*ScalarCpl*FG[-1])

        dFdt[-1,1] = (bdrF[-1,1] - (4+x.ntr-1)*dlnkhdt*FB[-1] + 2*aAlpha*scale*FG[-2]) 

        dFdt[-1,2] = (bdrF[-1,2] - (4+x.ntr-1)*dlnkhdt*FG[-1] + aAlpha*scale*(FE[-2] - FB[-2]) + ScalarCpl*FB[-1])

        return dFdt
            
    #Run GEF
    def InitialiseGEF(x):
        yini = np.zeros((x.ntr*3+4))

        #ini is always in Planck units
        x.vals = x.ini.copy()
        
        if (x.units):
            x.f = x.Mpl
            x.omega = x.H0
            x.units = False
        else:
            x.f = 1.
            x.omega = 1.
            x.units = True
        
        yini[0] = 0
        yini[1] = x.ini["phi"]/x.f
        yini[2] = x.ini["dphi"]/(x.f*x.omega)
        yini[3] = np.log(abs(yini[2]*x.dIdphi()))
        
        x.f = x.Mpl
        x.omega = x.H0
        x.ratio = x.omega/x.f
        x.units = False
    
        return yini
    
    def TimeStep(x, t, y, rtol=1e-6):
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

        x.vals["kh"] = np.exp(y[3])

        F = y[4:]
        x.vals["F"] = F.reshape(x.ntr, 3)

        x.vals["E"] = x.vals["F"][0,0]*np.exp(4*(y[3]-y[0]))
        x.vals["B"] = x.vals["F"][0,1]*np.exp(4*(y[3]-y[0]))
        x.vals["G"] = x.vals["F"][0,2]*np.exp(4*(y[3]-y[0]))

        x.vals["H"] = x.FriedmannEq()
        x.vals["Hprime"] = x.FriedmannEq2()
        x.vals["xi"] = x.GetXi()

        return
    
    @AddEventFlags("End of inflation", True, 1, True)
    def __EndOfInflation__(x, t, y):
        dphi = y[2]
        V = x.V(x.f*y[1])/(x.f*x.omega)**2
        rhoEB = 0.5*(y[4]+y[5])*x.ratio**2*np.exp(4*(y[3]-y[0]))
        val = (dphi**2 - V + rhoEB)
        return val
    
    def IncreaseNtr(x, val=10):
        x.ntr+=val
        print(f"Increasing ntr by {val} to {x.ntr}.")
        return
    
    def SetupSolver(x, events):
        yvals = []
        tvals = []
        t_events = [[] for event in events]
        N_events = [[] for event in events]
        nfevs = []
        yini = x.InitialiseGEF()
        t0 = 0.
        return t0, yini, nfevs, yvals, tvals, t_events, N_events


    def SolveGEF(x, tend=120., atol=1e-20, rtol=1e-6, reachNend=True, Ntol=-1):
        mp.dps = 8
        t = Timer()

        events = []
        eventnames = []
        if reachNend: 
            events.append(x.__EndOfInflation__)
            eventnames.append(x.__EndOfInflation__.name)

        t0, yini, nfevs, yvals, tvals, t_events, N_events = x.SetupSolver(events)
        
        ODE = lambda t, y: x.TimeStep(t, y, rtol)

        t.start()

        done=False
        attempts = 0
        
        print(f"Attempting first run with ntr={x.ntr}")
        while not(done) and attempts<25:
            attempts += 1
            teval = np.arange(10*t0, 10*tend +1)/10 #hotfix to ensure teval[-1] <= tend
            try:
                sol = solve_ivp(ODE, [t0,tend], yini, t_eval=teval,
                                 method="RK45", atol=atol, rtol=rtol, events=events)
                assert sol.success
            except ValueError or AssertionError:
                print(f"The run failed at t={x.vals['t']}, N={x.vals['N']}.")

                x.IncreaseNtr(10)
                t0, yini, nfevs, yvals, tvals, t_events, N_events = x.SetupSolver(events)
            except RuntimeError:
                raise RuntimeError
            else:
                if reachNend:
                    if len(tvals)==0:
                        yvals.append(sol.y)
                        tvals.append(sol.t)
                    else:
                        #Do not double-count end-points of sub-solutions.
                        yvals.append(sol.y[:,1:])
                        tvals.append(sol.t[1:])
                    nfevs.append(sol.nfev)

                    try: 
                        NendNew = sol.y_events[0][0,0]
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
                        for i, event in enumerate(events):
                            t_events[i].append(sol.t_events[i])
                            N_events[i].append(sol.y_events[i][:,0])
                        print(f"The end of inflation was reached at t={np.round(sol.t_events[i][-1], 1)} and N={np.round(NendNew, abs(Ntol))}.")
                        if np.log10(abs(NendNew-x.Nend)) < Ntol:
                            done=True
                        else:
                            print(f"To ensure a convergent run, check stability against increasing ntr.")
                            x.IncreaseNtr(10)

                            Ninc = abs(NendNew - x.Nend)
                            tdiff = np.round(Ninc/x.vals["H"])
                            tend = min(np.round(sol.t[-1] + tdiff, 1), tend)
                            print(f"The solver aims at reaching t={tend}")
                            
                            t0, yini, nfevs, yvals, tvals, t_events, N_events = x.SetupSolver(events)

                            done=False
                        x.Nend = NendNew
                else:
                    yvals.append(sol.y)
                    tvals.append(sol.t)
                    done=True
            
        t.stop()

        sol.t = np.concatenate(tvals)
        sol.y = np.concatenate(yvals, axis=1)
        
        sol.attempts = attempts
        if attempts>1:
            sol.nfev = nfevs
        
        eventdic = dict(zip(eventnames, [{"t":None, "N":None} for event in eventnames]))
        for i, event in enumerate(eventdic.keys()):

            eventdic[event]["t"] = np.round(np.concatenate(t_events[i]), 1)
            eventdic[event]["N"] = np.round(np.concatenate(N_events[i]), 3)

        sol.events = eventdic

        return sol, done
        
    def WriteOutGEFResults(x, sol):
        t = sol.t
        y = sol.y
        parsold = list(x.vals.keys())
        parsold.remove("F")
        newpars = ["ddphi", "dlnkh"] #"E1", "B1", "G1", "Edot", "Bdot", "Gdot", "EdotBdr", "BdotBdr", "GdotBdr"]
        pars = parsold + newpars
        res = dict(zip(pars, [[] for par in pars]))
        for i in range(len(t)):
            x.DefineDictionary(t[i], y[:,i])
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


    def RunGEF(x, ntr, tend=120., atol=1e-20, rtol=1e-6, reachNend=True, printstats=False, Ntol=-1):
        x.ntr = ntr+1
        if not(x.completed):
            try:
                sol, done = x.SolveGEF(tend, atol=atol, rtol=rtol, reachNend=reachNend, Ntol=Ntol)
                
                if printstats:
                    PrintSol(sol)
                if sol.attempts >= 10 and not(done):
                    print(f"The run did not finish after {sol.attempts} attempts. Check the output for more information.")
                x.completed = done
                x.WriteOutGEFResults(sol)
            except TruncationError:
                print("A truncation error occured")
                sol = None
            except:
                raise RuntimeError    
            return sol
        else:
            print("This run is already completed, access data using GEF.vals")
            return
        
    def SaveData(x):
        if (x.completed):
            #x.Unitful()
            #Data is always stored without units
            if x.GEFData==None:
                path = f"./Out/GEF_Beta{x.beta}.dat"
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
        
        names = ["t", "phi", "dphi", "ddphi", "a", "H",
                 "E", "B", "G", "kh", "dlnkh"]# "E1", "B1", "G1", "Edot", "Bdot", "Gdot",]
        #Check if data file is in order:
        for name in names:
            if name not in data.keys():
                print("The file you provided does not contain information on the parameter " + name + ". Please provide a complete data file")
                print("A complete file contains information on the parameters:", names)
                raise ImportError
        
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

        """Bdrnames = ["EdotBdr","BdotBdr","GdotBdr"]
        for bdrname in Bdrnames:
            if bdrname not in data.keys():
                x.vals["EdotBdr"] = data["Edot"]
                x.vals["BdotBdr"] = data["Bdot"]
                x.vals["GdotBdr"] = data["Gdot"]"""

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
            x.vals["dlnkh"] = x.vals["dlnkh"]/omega
            """x.vals["E1"] = x.vals["E1"]/(omega)**5
            x.vals["B1"] = x.vals["B1"]/(omega)**5
            x.vals["G1"] = x.vals["G1"]/(omega)**5
            x.vals["Edot"] = x.vals["Edot"]/(omega)**5
            x.vals["Bdot"] = x.vals["Bdot"]/(omega)**5
            x.vals["Gdot"] = x.vals["Gdot"]/(omega)**5
            x.vals["EdotBdr"] = x.vals["EdotBdr"]/(omega)**5
            x.vals["BdotBdr"] = x.vals["BdotBdr"]/(omega)**5
            x.vals["GdotBdr"] = x.vals["GdotBdr"]/(omega)**5"""
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
            x.vals["dlnkh"] = x.vals["dlnkh"]*omega
            """x.vals["E1"] = x.vals["E1"]*(omega)**5
            x.vals["B1"] = x.vals["B1"]*(omega)**5
            x.vals["G1"] = x.vals["G1"]*(omega)**5
            x.vals["Edot"] = x.vals["Edot"]*(omega)**5
            x.vals["Bdot"] = x.vals["Bdot"]*(omega)**5
            x.vals["Gdot"] = x.vals["Gdot"]*(omega)**5
            x.vals["EdotBdr"] = x.vals["EdotBdr"]*(omega)**5
            x.vals["BdotBdr"] = x.vals["BdotBdr"]*(omega)**5
            x.vals["GdotBdr"] = x.vals["GdotBdr"]*(omega)**5"""
            x.omega = 1.
            x.f = 1.
            x.ratio = 1.
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
        f = CubicSpline(N, (dphi**2 - V + 0.5*(E+B)*x.omega**2/x.f**2))
        res = fsolve(f, max(N), 1e-4)[0]
        
        x.Nend = res

        if unitswereon:
            x.Unitful()
        return res