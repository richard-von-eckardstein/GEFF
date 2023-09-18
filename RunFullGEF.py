import pandas as pd
from utilities import GetPhysQuantities
from wrapper import *
from EoM import GetXi, FriedmannEq
import os
import sys
import optparse
import numpy as np
from scipy.integrate import solve_ivp

alpha = 0
Mpl = 1.

class GEF:
    def __init__(self, beta, Mpl, phi0, dphidt0, M, ntr, SE, approx=False):
        self.beta = beta
        self.f = Mpl
        self.mass = M
        self.SE = SE
        if (SE==None):
            self.ODE = self.fullGEF_NoSE()
        else:
            if (SE=="mix"):
                self.ODE = self.fullGEF_SE_mixed()
                self.conductivity = self.ComputeImprovedSigma()
            elif (-1. <= SE <=1.):
                self.ODE = self.fullGEF_SE_collinear(SE)
                self.conductivity = self.ComputeSigmaCollinear()
            else:
                print(SE, "is not a valid choice for SE")
        self.approx = approx
        self.ntr = ntr
        V = 0.5*phi0**2*M**2
        self.omega = FriedmannEq(1.0, dphidt0, V, 0., 0., 0., 1.0)
        self.ratio = omega/f
    
    def potential(self, x):
        return 0.5*(x.phi)**2 * self.mass**2

    def dIdphi(self, x):
        return self.beta/self.Mpl

    def ddIddphi(self, x):
        return 0.

    def dVdphi(self, x):
        return x.phi * self.mass**2
    
    def GetXi(self, x):
        return (x.dI * x.dphidt)/(2 * x.H)
    
    def GetS(self, x):
        return (a**(self.alpha) * x.sigmaE)/(2* x.H)
        
    def EoMphi(self, x):
        #Scalar Field EoM
        dscdt = np.zeros(2)
    
        dscdt[0] = x.dphidt
        dscdt[1] = (self.alpha-3)* x.H * x.dphidt - x.a**(2*self.alpha)*x.dV - x.a**(2*self.alpha)*x.dI*x.G[0]*self.ratio**2

        return dscdt
    
    def FriedmannEq(self, x):
        #Friedmann Equation
        Hsq = (1/3) * (0.5 * x.dphidt**2 + x.a**(2*self.alpha)*(x.V + self.ratio**2*(0.5*(x.E[0]+x.B[0]) + x.rhoChi)))
        return Hsq

    def BoundaryComputations(self, x):
        xieff = x.xi + x.sB
        
        

class vals:
    def __init__(vals, t, y, GEF):
        self.units = False
        self.t = t
        self.N = y[0]
        self.a = np.exp(vals.N)
        self.phi = y[1]
        self.dphidt = y[2]
        self.ddphiddt = GEF.EoMphi(self)[1]
        self.kh = np.exp(sol.y[3,:])
        self.V = GEF.potential(GEF.f*self.phi)/(GEF.f*GEF.omega)**2
        self.dV = GEF.dVdphi(GEF.f*self.phi)/(GEF.f*GEF.omega**2)
        self.dI = GEF.dIdphi(GEF.f*self.phi)*GEF.f
        self.ddI = GEF.ddIddphi(GEF.f*self.phi)*GEF.f**2
        if (GEF.SE == None):
            F = y[4:]
            self.rhoChi = 0.
            self.delta = 1.
            self.sigmaE = 0.
            self.sigmaB = 0.
        else:
            self.delta = y[4]
            self.rhoChi = y[5]
            F = y[6:]
            F.reshape(GEF.ntr, 3)
            self.sigmaE = GEF.conductivity()
            
        self.E = F[:,0]
        self.B = F[:,1]
        self.G = F[:,2]
        
        sigmaE = GEF.conductivities()
            
        H = np.sqrt(GEF.Friedmann(self))
        xi = GetXi(
        
        
        
        
    
        
        
        

def SetupFullGEF(beta, Mpl, phi0, dphidt0, M, ntr, SE=None, approx=False):
    #all quantities are assumed to be in Planck Units
    
    rhoChi0 = 0.
    Delta0 = 1.

    E0 = 0
    B0 = 0

    def potential(phi):
        return 0.5*phi**2 * M**2

    def dIdphi(phi):
        return beta/Mpl

    def ddIddphi(phi):
        return 0

    def dVdphi(phi):
        return phi * M**2
    
    H = FriedmannEq(1.0, dphidt0, potential(phi0), E0, B0, rhoChi0, 1.0, 1.0)
    xi = GetXi(dphidt0, dIdphi(phi0), H)
    
    omega = H
    f = Mpl
    
    yini = np.zeros(ntr*3 + 6)
    yini[0] = 0
    yini[1] = phi0/f
    yini[2] = dphidt0/(f*omega)
    yini[3] = np.log(2*abs(xi))
    if SE == None:
        yini = yini[:-2]
        func = lambda t, x: fullGEF_NoSE(t, x, f=f, omega=omega, approx=approx)
    else:
        yini[4] = Delta0
        yini[5] = rhoChi0
        if SE=="mix":
            func = lambda t, x: fullGEF_SE_mixed(t, x, f=f, omega=omega, approx=approx)
        elif (-1. <= SE <=1.):
            func = lambda t, x: fullGEF_SE_collinear(t, x, f=f, omega=omega, frac=SE, approx=approx)
        else:
            print(SE, "is not a valid choice for SE")
            return
 
    return func, yini, potential, dVdphi, dIdphi, ddIddphi, omega, f

def get_cmdline_arguments():
    """
        Returns dictionary of command line arguments supplied to PhonoDark.
    """

    parser = optparse.OptionParser()
    parser.add_option('-n', action="store", default=1.,
            help="Truncation Number of GEF")
    parser.add_option('-b', action="store", default=10.,
            help="Coupling parameter beta")
    parser.add_option('-m', action="store", default=6e-6,
            help="Mass of the Inflaton in Mpl")
    parser.add_option('-p', action="store", default=15.55,
            help="Initial Field Value in Mpl")
    parser.add_option('-d', action="store", default=-np.sqrt(2/3),
            help="Initial Field Velocity in M*Mpl")
    parser.add_option('-s', action="store", default=None,
            help="SE Mode")
    parser.add_option('-a', action="store", default=True,
            help="approximate Whittaker Functions") 

    options_in, args = parser.parse_args()

    options = vars(options_in)

    return options

inopt = get_cmdline_arguments()

ntr = int(inopt.get("n"))
beta = float(inopt.get("b"))
M = float(inopt.get("m"))
phi0 = float(inopt.get("p")*Mpl)
dphidt0 = float(inopt.get("d")*M*Mpl)
SE = inopt.get("s")
approx = bool(inopt.get("a"))
if (type(SE) is str):
    if ("frac" in SE):
        SE = float(SE.replace("frac", ""))
        
o1 = GEF(beta, phi0, dphidt0, M, ntr, SE, False)
        
"""func, yini, potential, dVdphi, dIdphi, ddIddphi, omega, f = SetupFullGEF(beta, Mpl, phi0, dphidt0, M, ntr, SE=SE, approx=approx)

sol = solve_ivp(func, [0, 120] , yini, method="RK45")

if (SE == None):
    N, a, phi, dphidt, kh, E, B, G, V, H, xi = GetPhysQuantities(sol, beta, omega, f, SE=SE, units=False)
    data = [sol.t, N, a, H, kh, phi, dphidt, xi, V, E, B, G]

    names = ["t","N","a","H","kh","phi","dphidt","xi","V","E","B","G"]
else:
    N, a, phi, dphidt, kh, delta, rhoChi, E, B, G, V, H, xi, sigmaE, sigmaB = GetPhysQuantities(sol, beta, omega, f, SE=SE, units=False)
    data = [sol.t, N, a, H, kh, phi, dphidt, xi, V, E, B, G, delta, rhoChi, sigmaE, sigmaB]

    names = ["t","N","a","H","kh","phi","dphidt","xi","V","E","B","G","delta","rhoChi","sigmaE","sigmaB"]
    
DataDic = dict(zip(names, data))

filename = "GEF_Beta_" + str(beta) + "SE_" + str(SE) + ".dat"

DirName = os.getcwd() + "/Out/"

path = os.path.join(DirName, filename)

output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)"""



        

