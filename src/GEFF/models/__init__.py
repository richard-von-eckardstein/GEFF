"""
This module contains pre-defined GEF models.

Currently, the following models are implemented:
1. **'classic'**: pure axion inflation
2. **'SE noscale'**: fermionic axion inflation (without accounting for fermion $k$-dependence)
3. **'SE-kh'**: fermionic axion inflation (accounting for fermion $k$-dependence using $k_h$)

Execute these models by calling `GEFF.gef.GEF` with its name and settings.
"""
import importlib

from GEFF._docs import docs_models

#currently does not work...
module = importlib.import_module("GEFF.models.SE_kh")
for name, docs in docs_models.DOCS.items():
    getattr(module, name).__doc__ = docs

