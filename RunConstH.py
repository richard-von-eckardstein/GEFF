import numpy as np
from scipy.integrate import solve_ivp
import ConstH
import pandas as pd
import os
import Input as inpt
import sys

inopt, cmd_ok = inpt.get_cmdline_arguments()

print(inopt)
if not cmd_ok:
    print("ERROR")
    print("Need to give -x and -i as input")
    sys.exit()
    
outdir = "/scratch/tmp/rfreiher/GEF"

x = inopt.get("x")
b = inopt.get("i")
d = inopt.get("d")
s = inopt.get("s")
x = x.replace(",",".")
b = b.replace(",",".")
d = d.replace(",",".")
xi = float(x)
beta=10**float(b)
a = 1
ntr = 180
HConst = 1
MPl = 0.5e7*(beta/100)**(1/2)*np.exp(2.85*(xi-7))*HConst
I = beta/MPl
d = float(d)
s = float(s + "1")

print(xi, I, d, s)

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

DirName = "Out_xi" + x + "_beta" + b + "/"
DirName = os.path.join(outdir, DirName)

if (not os.path.exists(DirName)):
        os.mkdir(DirName)

filename = "Out_xi" + x + "_base" + "_beta" + b + ".dat"

path = os.path.join(DirName, filename)
        
output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)

dev = np.arange(1, d+1)
delta = s*1/10**(dev)
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
    
    filename = "Out_xi" + x + "_del" + str(s) + "_" + str(dev[j]) + "_beta" + b + "180.dat"

    path = os.path.join(DirName, filename)

    output_df = pd.DataFrame(DataDic)  
    output_df.to_csv(path)
