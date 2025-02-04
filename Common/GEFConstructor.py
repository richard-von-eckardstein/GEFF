import numpy as np
#from ..GEFClassic.GEFClassic import GEF as GEFClassic
#from ..GEFSchwinger.GEFSchwinger import GEF as GEFSchwinger
import importlib.util as util
from types import ModuleType
import logging

log = logging.getLogger()

def load_GEFConfig(GEFFile: str):
    #Load Attributes from GEFFile
    spec = util.spec_from_file_location("GEFInput", GEFFile)
    mod  = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod

def check_GEFConfig(GEFconfig: ModuleType):
    #Check if necessary input is given, if not, raise an error
    necessaryInput = ["beta", "V", "dVdphi", "init"]

    for key in necessaryInput:
        if not hasattr(GEFconfig, key):
            error = f"{key} is a necessary input for the GEF."
            log.error(error)
            raise SystemExit

    #Check if Optional Input is given, if not, set defaults
    defaultOptionalInput = {
        "SE" : False,
        "SEMode" : None,
        "SEPicture" : None,
        "GEFFile" : None,
        "ModeFile" : None
    }

    for key in defaultOptionalInput.keys():
        if not hasattr(GEFconfig, key):
            setattr(GEFconfig, key, defaultOptionalInput[key])
            message = f"{key}  is not found in the given file. Seeting {key} to {defaultOptionalInput[key]}"
            log.info(message)
    
    #Check individual entries for consistency
    if not isinstance(GEFconfig.beta, int):
        error = f"The coupling strength beta must be an integer."
        log.error(error)
        raise SystemExit
    
    if not isinstance(GEFconfig.V, function):
        err = True
    else:
        if isinstance(GEFconfig.V(1), int):
            err = True

    if err:
        error = f"The potential must be a real function of phi."
        log.error(error)
        raise SystemExit


    return