"""
This module contains pre-defined GEF models.

Currently, the following models are implemented:
1. **'classic'**: pure axion inflation
2. **'SE noscale'**: fermionic axion inflation (without accounting for fermion $k$-dependence)
3. **'SE-kh'**: fermionic axion inflation (accounting for fermion $k$-dependence using $k_h$)

Execute these models by calling `GEFF.gef.GEF` with its name and settings.
"""
import importlib
import os

def load_model(model : str):
    """
    Import and execute a module defining a GEF model.

    Parameters
    ----------
    model : str
        The name of the GEF model or a full dotted import path (e.g., "path.to.module").
    settings : dict
        A dictionary containing updated settings for the module.

    Returns
    -------
    ModuleType
        The configured module.
    """

    # Case 1: Bare name, resolve to ./models/{name}.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelpath = os.path.join(current_dir, f"{model}.py")

    if os.path.exists(modelpath):
        spec = importlib.util.spec_from_file_location(model, modelpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    else:
        # Case 2: Try treating it as a dotted import path
        try:
            mod = importlib.import_module(model)
        except ImportError as e:
            raise FileNotFoundError(
                f"No model file found at '{modelpath}' and failed to import '{model}'"
                ) from e
    return mod