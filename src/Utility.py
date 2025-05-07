import natpy as nat
import numpy as np
import pandas as pd
import os

basepath = os.path.dirname(os.path.abspath(__file__))

nat.set_active_units("HEP")

#constants
G = 6.67430e-11 * nat.convert(nat.m**3 * nat.kg**(-1) * nat.s**(-2), nat.GeV**(-2)) #Newtons constant in GeV (from https://pdg.lbl.gov/2024/reviews/rpp2024-rev-phys-constants.pdf)
M_pl = (8*np.pi*G)**(-1/2) # reduced Planck mass in GeV

#Planck results
T_0 = 2.7255 * nat.convert(nat.K, nat.GeV) # DOI 10.1088/0004-637X/707/2/916
h = 0.674 #reduced Hubble parameter https://arxiv.org/abs/1807.06209


#Cosmological quantities
gtab = pd.read_table(os.path.join(basepath, "Data/EffDoF.dat"), sep=" ", comment="#").values.T
gdic = {"T":gtab[0,:],"f":gtab[1,:],"grho":gtab[2,:], "gs":gtab[3,:]}

def g_s(T):
    return np.interp(T, gdic["T"], gdic["gs"])

def g_rho(T):
    return np.interp(T, gdic["T"], gdic["grho"])

def g_s_freq(f):
    return np.interp(f, gdic["f"], gdic["gs"])

def g_rho_freq(f):
    return np.interp(f, gdic["f"], gdic["grho"])

g_rho_0 = g_rho(T_0)
g_s_0 = g_s(T_0)

H0 = h*100*nat.convert(nat.km * nat.s**(-1) * nat.Mpc**(-1), nat.GeV)
omega_r = np.pi**2*g_rho_0/(90*M_pl**2*H0**2)*T_0**4

#Conversions
gev_to_hz = nat.convert(nat.GeV, nat.Hz)

