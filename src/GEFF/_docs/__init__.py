def assign_docstrings(module, docs):
    for func_name, docstring in docs.items():
        if hasattr(module, func_name) and callable(getattr(module, func_name)):
            getattr(module, func_name).__doc__ = docstring
    return module