import numpy as np
from scipy.integrate import solve_ivp
import ConstH
import pandas as pd
import os
import Input as inpt
import sys

inopt, cmd_ok = inpt.get_cmdline_arguments()

if not cmd_ok:
    print("ERROR")
    print("Need to give -x and -i as input")
    sys.exit()
    
print(inopt)
    
outdir = "/scratch/tmp/rfreiher/GEF"
x = inopt.get("x")
i = inopt.get("i")
d = inopt.get("d")
xi = float(x)
a = 1
ntr = 190
HConst = 1
I = float(i)
d = float(d)

print(xi, I, d)

yini, dVini = ConstH.SetupConstH(x, HConst, a, ntr, I)

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

filename = "Out_xi" + x + "_base" + "_I" + i + ".dat"
print(filename)
path = os.path.join(outdir, filename)
        
output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)

dev = np.arange(1, d+1)
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
    
    filename = "Out_xi" + x + "_del" + str(dev[j]) + "_I" + i + ".dat"
    path = os.path.join(outdir, filename)

    output_df = pd.DataFrame(DataDic)  
    output_df.to_csv(path)
