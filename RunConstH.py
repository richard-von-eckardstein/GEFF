import numpy as np
from scipy.integrate import solve_ivp
import ConstH
import pandas as pd
import os

def getNCut(t, xi, xi0, prec=10**(-0.5)):
    for i in range(len(xi)):
        diff = abs(xi[i]-xi0)/xi0
        if (diff > prec):
            return t[i]
    return False

def NCutbyDelta(f, t, yini, vals, prec=10**(-0.5)):
    xi0 = yini[0]
    xiend = (1+prec)*xi0
    xis = np.linspace(xi0, xiend, vals+1)
    NCut = []
    delta = []
    for i in range(vals):
        yini[0] = xis[i]
        sol = solve_ivp(f, [t[0], t[-1]] , yini)
        NCut.append(getNCut(sol.t, sol.y[0,:], xi0, prec=10**(-0.5)))
        delta.append(abs(xis[i]-xi0))
            
    return delta, NCut

outdir = "/home/richard/Documents/Phd Muenster/Axions in the early Universe"

xi = 7
a = 1
ntr = 190
HConst = 1
I = 2.45e-5
yini, dVini = ConstH.SetupConstH(xi, HConst, a, ntr, I)

steps = int(1e6)
N = np.linspace(0, 50, steps)
t = N/HConst

CH = lambda t, x: ConstH.ConstHGEF(x, t, HConst, dVini, I)

delta, NCut = NCutbyDelta(CH, t, yini, 2)

DataDic = dict(d = delta, N = NCut)

filename = "NCut_xi_" + str(xi)
path = os.path.join(outdir, filename)
        
output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)
