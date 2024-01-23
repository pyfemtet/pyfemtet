class ModelError(Exception):
    """Exception raised for errors in the model update.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


class MeshError(Exception):
    """Exception raised for errors in the meshing.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


class SolveError(Exception):
    """Exception raised for errors in the solver.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


# class PostError(Exception):
#     pass


# class FEMCrash(Exception):
#     pass


class FemtetAutomationError(Exception):
    """Exception raised for errors in automating Femtet."""
    pass
