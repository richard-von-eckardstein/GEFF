#Space--time variables:
spacetime = {
    "t": {"H0":-1, "MP":0, "default":0.}, #physical time
    #"eta": {"H0":-1, "MP":0, "default":0}, #conformal time
    "N": {"H0":0, "MP":0, "default":0.}, #e-folds
    "a": {"H0":0, "MP":0, "default":1.}, #scale factor
    "H": {"H0":1, "MP":0} #Hubble rate
}


#Inflaton  variables:
inflaton = {
    "phi": {"H0":0, "MP":1}, #field value
    "dphi": {"H0":1, "MP":1}, #velocity
    "ddphi": {"H0":2, "MP":1} #acceleration
}

#Gauge-field variables:
gaugefield = {
    "E": {"H0":4, "MP":1, "default":0.}, #electric field expectation value
    "B": {"H0":4, "MP":1, "default":0.}, #magnetic field expectation value
    "G": {"H0":4, "MP":1, "default":0.} #-EdotB expectation value
}

#Auxiliary quantities:
auxiliary = {
    "xi": {"H0":0, "MP":0}, #instability parameter
    "kh": {"H0":1, "MP":0}  #instability scale
}

#Inflaton potential:
inflatonpotential = {
                    "V":{"H0":2, "MP":2},
                    "dV":{"H0":2, "MP":1},
                    }

#Inflaton--gauge-field coupling:
coupling = {
            "dI":{"H0":0, "MP":-1},
            "ddI":{"H0":0, "MP":-2},
            }