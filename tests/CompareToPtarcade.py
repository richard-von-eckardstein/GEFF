from ptarcade import models_utils
from src import Utility

def CompareResults(val):
    v1 = getattr(models_utils, val)
    v2 = getattr(Utility, val)
    if callable(v1) and callable(v2):
        return CompareTwoFuncs(v1, v2)
    else:
        return CompareTwoConstants(v1, v2)

def CompareTwoConstants(v1, v2):
    def CallableCompare(*args):
        return 1-v2/v1
    return CallableCompare

def CompareTwoFuncs(v1, v2):
    def CallableCompare(*args):
        x = v1(*args)
        y = v2(*args)
        return 1-x/y
    return CallableCompare

vals = ["g_rho", "g_rho_0", "g_s", "g_s_0", "T_0", "M_pl", "gev_to_hz", "h", "omega_r"]

testvals = [1e-5, 1.541, 1.6809e5, 8.121e17, Utility.M_pl]
for val in vals:
    print(val)
    for test in testvals:
        print("test",test, ": diff", CompareResults(val)(test))

freqs = [1e-17, 6.23861e-12, 2.2312e-6, 1.23123e3, 9.12313e5]
print("g_rho")
for freq in freqs:
    print("test",freq, ": diff", 1 - Utility.g_rho_freq(freq)/models_utils.g_rho(freq, True))
print("g_s")
for freq in freqs:
    print("test",freq, ": diff", 1 - Utility.g_s_freq(freq)/models_utils.g_s(freq, True))


    

