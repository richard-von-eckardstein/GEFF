DOCS = {
    "module":"""
    This module defines the main `BaseGEF` class and its class factory `GEF`.
    """,
    "BaseGEF":"""
    This class is the primary interface of a GEF model.

    The class contains a `GEFSolver` and a `ModeSolver` that are used to solve GEF equations in `run`.

    The class also stores the evolution of the GEF-model variables as obtained from `run` or `load_GEFdata`.
    You can access these variables like a `.bgtypes.BGSystem`. E.g., you can access the $e$-folds variable through the attribute `N`.

    As a child of `BGSystem`, the GEFF can be passed to other tools initialized by `BGSystem`'s. For example, to compute the tensor power spectrum from your GEF solution,
    you can pass it to `.tools.pt.PT`.

    The `BaseGEF` contains the model `.models.classic`. To define a custom GEF model, use the class factory `GEF`.
    """
}
