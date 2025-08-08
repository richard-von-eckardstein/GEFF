from GEFF.BGTypes import BGVal, BGFunc

#Space--time variables:
t=BGVal("t", -1, 0) #physical time
N=BGVal("N", 0, 0) #e-folds
a=BGVal("a", 0, 0) #scale factor
H=BGVal("H", 1, 0) #Hubble rate
spacetime = {t, N, a, H}

#Inflaton  variables:
phi=BGVal("phi", 0, 1) #inflaton field
dphi=BGVal("dphi", 1, 1) #inflaton velocity
ddphi=BGVal("ddphi", 2, 1) #inflaton acceleration
inflaton = {phi, dphi, ddphi}

#Inflaton potential
V=BGFunc("V", [phi], 2, 2) #scalar potential
dV=BGFunc("dV", [phi], 2, 2) #scalar-potential derivative
inflatonpotential={V, dV}

#Inflaton--gauge-field coupling:
dI=BGFunc("dI", [phi], 0, -1) #scalar potential
ddI=BGFunc("ddI", [phi], 0, -2) #scalar-potential derivative
coupling = {dI, ddI}

#Gauge-field variables:
E=BGVal("E", 4, 0) #electric field expectation value
B=BGVal("B", 4, 0) #magnetic field expectation value
G=BGVal("G", 4, 0) #-EdotB expectation value
gaugefield = {E, B, G}

#Auxiliary quantities:
xi=BGVal("xi", 0, 0) #instability parameter
kh=BGVal("kh", 1, 0) #instability scale
auxiliary = {xi, kh}