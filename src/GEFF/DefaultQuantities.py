from GEFF.BGTypes import BGVal, BGFunc

#Space--time variables:
t=BGVal("t", -1, 0) #physical time
N=BGVal("N", 0, 0) #e-folds
a=BGVal("a", 0, 0) #scale factor
H=BGVal("H", 1, 0) #Hubble rate

#Inflaton  variables:
phi=BGVal("phi", 0, 1) #inflaton field
dphi=BGVal("dphi", 1, 1) #inflaton velocity
ddphi=BGVal("ddphi", 2, 1) #inflaton acceleration

#Inflaton potential
V=BGFunc("V", [phi], 2, 2) #scalar potential
dV=BGFunc("dV", [phi], 2, 2) #scalar-potential derivative

#Gauge-field variables:
E=BGVal("E", 4, 0) #electric field expectation value
B=BGVal("B", 4, 0) #magnetic field expectation value
G=BGVal("G", 4, 0) #-EdotB expectation value

#Auxiliary quantities:
xi=BGVal("xi", 0, 0) #instability parameter
kh=BGVal("kh", 1, 0) #instability scale

#constants
beta=BGVal("beta", 0, -1) #inflaton--gauge-field coupling beta/Mp