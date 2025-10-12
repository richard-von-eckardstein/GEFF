DOCS = {
    "module":"""
    This module defines the `BaseGEF` class and its class factory `compile_model`.
    """,
    "BaseGEF":"""
    This class defines the basic properties of a GEF model.

    Each GEF model contains a `GEFSolver` and a `ModeSolver` that are used to solve GEF equations with the `run` method.

    Instances of a GEF model can also be used to load data using `load_GEFdata`.
     Loading data this way creates a `BGSystem` which also knows the appropriate units, constants and functions for this model.

    The `BaseGEF` is a compiled version of `.models.pai`. To compile other models, use the class factory `compile_model`.
    """
}
