import os
from src.GEFClassic.GEFClassic import GEF as GEFClassic
from src.GEFSchwinger.GEFSchwinger import GEF as GEFSchwinger
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
    necessaryInput = ["beta", "V", "dVdphi", "InitialConditions"]

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
        GEFconfig.V(1.)
        GEFconfig.dVdphi(1.)
    except TypeError:
        error = f"The potential and its derivative must be a function of the field phi"
        raise TypeError(error)
    except AssertionError:
        error = f"The potential and its derivative must return a real scalar."
        raise Exception(error)
    
    PhiKeys = ["phi", "dphi"]
    try:
        keys = GEFconfig.InitialConditions.keys()
        for PhiKey in PhiKeys:
            assert PhiKey in keys
            assert isinstance(GEFconfig.InitialConditions[PhiKey], float)
        warn = False
        for key in keys:
            if "E" in key or "B" in key or "G" in "key":
                warn = True
        if warn:
            warning = "\nGiving initial data for bilinear gauge field expectation values is not yet implemented. Keys in 'InitialConditions' refering to gauge-fields will be ignored."
            warnings.warn(warning)

    except AttributeError:
        error =  "'InitialConditions' must be a dictionary"
        raise AttributeError(error)
    except AssertionError:
        error = "'InitialConditions' must contain initial data for the inflaton field with keys 'phi' and 'dphi'"
        raise Exception(error)

    if not isinstance(GEFconfig.SE, bool):
        error =  "'SE' must be a boolean."
        raise Exception(error)
        
   
    if GEFconfig.SE:
        if not (GEFconfig.SEModel in ["KDep", "Del1", "Old"]):
            error =  f"'SEModel' must be 'KDep', 'Del1' or 'Old'"
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
        warning =  f"\n'MbMFile' must be a path to a file or None. Will be set to None."
        GEFconfig.MbMFile=None
        warnings.warn(warning)
    
    return GEFconfig

def CreateGEF(GEFFile: str, approx=True):
    #Load GEF configurations and check for consistency
    GEFconfig = load_GEFConfig(GEFFile)
    GEFconfig = check_GEFConfig(GEFconfig)

    #common input:
    Mpl = 1.

    beta = GEFconfig.beta

    V = GEFconfig.V
    dV = GEFconfig.dVdphi

    InitialConditions = GEFconfig.InitialConditions

    #Setup Schwinger GEF or Classic GEF depending on GEFconfig.SE
    if GEFconfig.SE:
        if GEFconfig.SEPicture=="electric":
            SE=-1
        elif GEFconfig.SEPicture=="magnetic":
            SE=1
        elif GEFconfig.SEPicture=="mixed":
            SE="mix"

        G = GEFSchwinger(beta, InitialConditions, V, dV, SEPicture=SE, SEModel=GEFconfig.SEModel, GEFData=GEFconfig.GEFFile, ModeData=GEFconfig.MbMFile, approx=approx)
    else:
        G = GEFClassic(beta, InitialConditions, V, dV, GEFData=GEFconfig.GEFFile, ModeData=GEFconfig.MbMFile, approx=approx)

    if GEFconfig.GEFFile==None:
        print("No existing GEF data found. You can specify a file to load using GEF.LoadData or solve the GEF equations using GEF.RunGEF.")
    else:
        try:
            G.LoadData()
            G.Unitful()
        except:
            print("Could not load GEF data from file.")

    return G


