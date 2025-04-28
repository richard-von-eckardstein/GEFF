import pandas as pd
import os
import sys
import optparse
import numpy as np
from scipy.integrate import solve_ivp
from GEF import GEF

alpha = 0
Mpl = 1.

def get_cmdline_arguments():
    #Returns dictionary of command line arguments supplied to PhonoDark.

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
    parser.add_option('-t', action="store", default=120,
            help="Target end time for GEF") 
    parser.add_option('--load', action="store", default=None,
            help="option to load existing GEF Run from file")
    parser.add_option('--save', action="store", default=None,
            help="option to load existing GEF Run from file")

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
file = inopt.get("l")
t = float(inopt.get("t"))
if (type(SE) is str):
    if ("frac" in SE):
        SE = float(SE.replace("frac", ""))
        
        
dic = {"phi":phi0, "dphi":dphidt0, "delta":1., "rhoChi":0.}
G = GEF(alpha, beta, Mpl, dic, M, ntr, SE, AltDamp=2, approx=True)

if file==None:
    G.RunGEF(t1=t)
else:
    G.LoadData(file)
    t = min(G.vals["t"][-1], t)
    G.IterateGEF(t1=t)
G.SaveData()

