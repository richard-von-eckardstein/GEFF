"""
This module contains pre-defined GEF models.

Currently, the following models are implemented:
1. **'pai'**: pure axion inflation
2. **'fai_basic'**: fermionic axion inflation (without accounting for fermion $k$-dependence)
3. **'fai_kh'**: fermionic axion inflation (accounting for fermion $k$-dependence using $k_h$)

Execute these models by calling `GEFF.gef.GEF` with its name and settings.
"""
import importlib
import os

def load_model(model : str, user_settings : dict={}):
    """
    Import and execute a module defining a GEF model.

    Parameters
    ----------
    model : str
        The name of the GEF model or a full dotted import path (e.g., "path.to.module").
    user_settings : dict
        A dictionary containing updated settings for the module.

    Returns
    -------
    ModuleType
        The configured module.
    """

    # if model is  astring, load  ./models/{name}.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"{model}.py")

    if os.path.exists(modelpath):
        spec = importlib.util.spec_from_file_location(model, modelpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    else:
        # if model is not a string, try loading it directly as a module
        try:
            mod = model
        except ImportError as e:
            raise FileNotFoundError(
                f"No model file found at '{modelpath}' and failed to import '{model}'"
                ) from e
        
    if hasattr(mod, "settings") and isinstance(user_settings, dict):
        for key, item in user_settings.items():
            if key in mod.settings:
                mod.settings[key] = item
                print(f"Updating '{key}' to '{item}'.")
            else:
                print(f"Ignoring unknown model setting '{key}'.")
        mod.interpret_settings()

    return mod