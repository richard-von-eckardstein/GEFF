DOCS = {
    "module":"""
    This module defines the main `BaseGEF` class and its class factory `GEF`.
    """,
    "BaseGEF":"""
    This class is the primary interface of a GEF model.

    The class contains a `GEFSolver` and a `ModeSolver`, which are used in solving the GEF equations in the method `run`.

    The class also stores the evolution of the GEF-model variables, which are either obtained through `run` or `load_GEFdata`.
    You can access these variables like a `GEFF.bgtypes.BGSystem`. E.g., you can access the evolution of the $e$-folds parameter in the attribute `N`.

    The `BaseGEF` contains the model `GEFF.models.classic`. To create a custom GEF model from a model file, use `GEF`.
    """
}