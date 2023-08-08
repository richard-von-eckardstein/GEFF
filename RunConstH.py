import numpy as np
from scipy.integrate import solve_ivp
from wrapper import ConstHGEF
import AS_Setup as AS
import pandas as pd
import os
import Input as inpt
import sys

inopt, cmd_ok = inpt.get_cmdline_arguments()

print(inopt)
if not cmd_ok:
    print("ERROR")
    print("Need to give -x and -b as input")
    sys.exit()
    
outdir = "/home/richard/Documents/Phd Muenster/Axions in the early Universe/DataTables"

x = inopt.get("x")
b = inopt.get("b")
d = inopt.get("d")
s = inopt.get("s")
x = x.replace(",",".")
b = b.replace(",",".")
d = d.replace(",",".")
xi = float(x)
beta=10**float(b)

if (d=="None"):
    d=None
else:
    d = float(d)
    s = float(s + "1")

a = 1
ntr = 180
HConst = 1

print(xi, beta, d, s)

Fvec, lnkh, _, dVini, I, f, omega = AS.SetupConstH(xi, beta, a, ntr)

yini = np.zeros((2+3*ntr))

yini[0] = xi*(1)
yini[1] = lnkh
yini[2:] = Fvec

steps = int(1e6)
N = np.linspace(0, 60, steps)
t = N/HConst

if (d==None):
    yini[0] = xi
else:
    delta = s*1/10**d
    yini[0] = (delta+1)*xi 

steps = int(1e6)
N = np.linspace(0, 60, steps)
t = N/HConst

CH = lambda t, x: ConstHGEF(x, t, HConst, dVini, I, omega, f, approx=False)

sol = solve_ivp(CH, [t[0], t[-1]] , yini)

data = [sol.t, sol.y[0], sol.y[1]]

names = ["t", "xi", "logkh"]
"""for i in range(ntr):
    data.append(sol.y[2+i*3])
    names.append("E(" + str(i) + ")")
    data.append(sol.y[3+i*3])
    names.append("B(" + str(i) + ")")
    data.append(sol.y[4+i*3])
    names.append("G(" + str(i) + ")")"""
    
DataDic = dict(zip(names, data))

if(d == None):
    filename = "Out_xi" + x + "_base" + "_beta" + b + ".dat"
else:
    filename = "Out_xi" + x + "_del" + str(s) + "_" + str(d) + "_beta" + b + ".dat"

DirName = ""

path = os.path.join(DirName, filename)

output_df = pd.DataFrame(DataDic)  
output_df.to_csv(path)
