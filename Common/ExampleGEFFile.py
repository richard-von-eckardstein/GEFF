

#Define the inflaton--gauge-field coupling strength beta/M_pl \phi F Ftilde
beta = 10

#Define Potentials (in Planck units)
m = 6e-6

V = lambda phi: 0.5*m**2*phi**2
dVdphi = lambda phi: m**2*phi

#configure initial data (in Planck units). 
# Necessary keys: "phi", "dphi". 
# Optional keys: "En", "Bn", "Gn" for integers n>=0 (Currently not implemented)
# If optional keys are not given, all gauge-field expectation values are assumed to be zero
phi0 = 15.55
dphidt0 = -(2/3)**(1/2)*m

init = {"phi":phi0, "dphi":dphidt0}


#Toggle Schwinger Effect (True/False)
SE=True

#Decide Schwinger Effect Treatment ("KDep", "Del1", None)
SEModel="KDep"

#Decide Schwinger Effect picture ("electric", "magnetic", "mixed")
SEPicture="mixed"

#Path to file containing available GEF data given the above input. If None, it is assumed that no GEF data exists.
GEFFile="be"#None

#Path to file containing available Mode-by-Mode data given the above input. If None, it is assumed that no Mode-by-Mode data exists.
MbMFile=None


