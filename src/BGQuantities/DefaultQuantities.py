from src.BGQuantities.BGTypes import DefineQuantity
#Space--time variables:
spacetime = {
    DefineQuantity("t", -1, 0), #physical time
    DefineQuantity("N", 0, 0), #e-folds
    DefineQuantity("a", 0, 0), #scale factor
    DefineQuantity("H", 1, 0) #Hubble rate
}


#Inflaton  variables:
inflaton = {
    DefineQuantity("phi", 0, 1), #inflaton field
    DefineQuantity("dphi", 1, 1), #inflaton velocity
    DefineQuantity("ddphi", 2, 1) #inflaton acceleration
}

#Gauge-field variables:
gaugefield = {
    DefineQuantity("E", 4, 0), #electric field expectation value
    DefineQuantity("B", 4, 0), #magnetic field expectation value
    DefineQuantity("G", 4, 0)#-EdotB expectation value
}

#Auxiliary quantities:
auxiliary = {
    DefineQuantity("xi", 0, 0), #instability parameter
    DefineQuantity("kh", 1, 0) #instability scale
}

#Inflaton potential:
inflatonpotential = {
    DefineQuantity("V", 2, 2, isfunc=True), #scalar potential
    DefineQuantity("dV", 2, 2, isfunc=True), #scalar-potential derivative
}

#Inflaton--gauge-field coupling:
coupling = {
    DefineQuantity("dI", 0, -1, isfunc=True), #scalar potential
    DefineQuantity("ddI", 0, -2, isfunc=True), #scalar-potential derivative
}