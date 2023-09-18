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
        
func, yini, potential, dVdphi, dIdphi, ddIddphi, omega, f = SetupFullGEF(beta, Mpl, phi0, dphidt0, M, ntr, SE=SE, approx=approx)

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
output_df.to_csv(path)



        

