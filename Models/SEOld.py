name = "Schwinger(Old)"

modelQuantities = {
                "sigmaE": {"H0":1, "MP":0, "optional":False}, #electric damping
                "sigmaB": {"H0":1, "MP":0, "optional":False}, #magnetic damping
                "delta": {"H0":0, "MP":0, "optional":False}, #integrated electric damping
                "xieff": {"H0":0, "MP":0}, #effective instability parameter
                "rhoChi": {"H0":4, "MP":0, "default":0., "optional":False} #Fermion energy density 
                }   

modelFunctions = {}

rhoFerm = lambda dic: dic["rhoChi"]["value"]
modelRhos = [rhoFerm]

modelSettings = {"picture": "mixed"}


