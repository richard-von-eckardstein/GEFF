import numpy as np
from scipy.integrate import solve_ivp
import ConstH
import pandas as pd
import os

outdir = "/scratch/tmp/rfreiher/GEF"

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

sol = solve_ivp(CH, [t[0], t[-1]] , yini)

data = [sol.t, sol.y[0], sol.y[1]]

names = ["t", "xi", "logkh"]
for i in range(ntr):
    data.append(sol.y[2+i*3])
    names.append("E(" + str(i) + ")")
    data.append(sol.y[3+i*3])
    names.append("B(" + str(i) + ")")
    data.append(sol.y[4+i*3])
    names.append("G(" + str(i) + ")")
    
DataDic = dict(zip(names, data))

filename = "Out_xi" + str(xi) + "_base" + "_I" + str(I) + ".dat"
path = os.path.join(outdir, filename)
        
output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)

dev = np.arange(1, 17)
delta = 1/10**(dev)
xis = (delta+1)*xi
for j in range(dev.size):
    yini[0] = xis[j]
    sol = solve_ivp(CH, [t[0], t[-1]] , yini)

    data = [sol.t, sol.y[0], sol.y[1]]

    names = ["t", "xi", "logkh"]
    for i in range(ntr):
        data.append(sol.y[2+i*3])
        names.append("E(" + str(i) + ")")
        data.append(sol.y[3+i*3])
        names.append("B(" + str(i) + ")")
        data.append(sol.y[4+i*3])
        names.append("G(" + str(i) + ")")

    DataDic = dict(zip(names, data))
    
    filename = "Out_xi" + str(xi) + "_del" + str(dev[j]) + "_I" + str(I) + ".dat"
    path = os.path.join(outdir, filename)

    output_df = pd.DataFrame(DataDic)  
    output_df.to_csv(path)
