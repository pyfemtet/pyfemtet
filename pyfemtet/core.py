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


def _version(
        main=None,
        major=None,
        minor=None,
        Femtet=None,
):
    if Femtet is not None:
        assert (main is None) and (major is None) and (minor is None), 'バージョンを指定しないでください'
        main, major, minor = [int(v) for v in Femtet.Version.split('.')[:3]]
    else:
        assert (main is not None) and (major is not None) and (minor is not None), 'バージョンを指定してください'
    return main*10000 + major*100 + minor
