import os
#from ..GEFClassic.GEFClassic import GEF as GEFClassic
#from ..GEFSchwinger.GEFSchwinger import GEF as GEFSchwinger
import importlib.util as util
from types import ModuleType
import warnings


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
            error = f"'{key}' is a necessary input for the GEF."
            raise Exception(error)

    #Check if Optional Input is given, if not, set defaults
    defaultOptionalInput = {
        "SE" : False,
        "SEModel" : None,
        "SEPicture" : None,
        "GEFFile" : None,
        "MbMFile" : None
    }

    for key in defaultOptionalInput.keys():
        if not hasattr(GEFconfig, key):
            setattr(GEFconfig, key, defaultOptionalInput[key])
  
    #Check individual entries for consistency
    if not (isinstance(GEFconfig.beta, float) or isinstance(GEFconfig.beta, int)):
        error = f"The coupling strength beta must be a real number."
        raise Exception(error)
    
    try:
        assert isinstance(GEFconfig.V(1.), float)
        assert isinstance(GEFconfig.dVdphi(1.), float)
    except TypeError:
        error = f"The potential and its derivative must be a function of the field phi"
        raise TypeError(error)
    except AssertionError:
        error = f"The potential and its derivative must return a real scalar."
        raise Exception(error)
    
    PhiKeys = ["phi", "dphi"]
    try:
        keys = GEFconfig.init.keys()
        for PhiKey in PhiKeys:
            assert PhiKey in keys
            assert isinstance(GEFconfig.init[PhiKey], float)
        warn = False
        for key in keys:
            if "E" in key or "B" in key or "G" in "key":
                warn = True
        if warn:
            warning = "\nGiving initial data for bilinear gauge field expectation values is not yet implemented. Keys in 'init' refering to gauge-fields will be ignored."
            warnings.warn(warning)

    except AttributeError:
        error =  "'init' must be a dictionary"
        raise AttributeError(error)
    except AssertionError:
        error = "'init' must contain initial data for the inflaton field with keys 'phi' and 'dphi'"
        raise Exception(error)

    if not isinstance(GEFconfig.SE, bool):
        error =  "'SE' must be a boolean."
        raise Exception(error)
        
   
    if GEFconfig.SE:
        if (not (GEFconfig.SEModel in ["KDep", "Del1"]) or (GEFconfig.SEModel==None)):
            error =  f"'SEModel' must be 'KDep', 'Del1' or None"
            raise Exception(error)
        
        if not (GEFconfig.SEPicture in ["electric", "magnetic", "mixed"]):
            error =  f"'SEPicture' must be 'electric', 'magnetic' or 'mixed'"
            raise Exception(error)
    
    if isinstance(GEFconfig.GEFFile, str):
        if not os.path.isfile(GEFconfig.GEFFile):
            warning =  f"\nNo file found under {GEFconfig.GEFFile}. This input will be ignored"
            GEFconfig.GEFFile=None
            warnings.warn(warning)
    elif not GEFconfig.GEFFile==None:
        warning =  f"\n'GEFFile' must be a path to a file or None. This input will be ignored"
        GEFconfig.GEFFile=None
        warnings.warn(warning)
        
    if isinstance(GEFconfig.MbMFile, str):
        if not os.path.isfile(GEFconfig.MbMFile):
            warning =  f"\nNo file found under {GEFconfig.MbMFile}. This input will be ignored"
            GEFconfig.MbMFile=None
            warnings.warn(warning)
    elif not GEFconfig.MbMFile==None:
        warning =  f"\n'GEFFile' must be a path to a file or None. This input will be ignored"
        GEFconfig.MbMFile=None
        warnings.warn(warning)
    
    return GEFconfig

