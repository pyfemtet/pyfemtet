try:
    import brepmatching
except ModuleNotFoundError as e:
    import warnings
    warnings.warn(
        'There is no installation of `brepmatching`. '
        'Please confirm installation via '
        '`pip install pyfemtet[brep]` or '
        '`pip install brepmatching` command.'
    )
    raise e

from brepmatching.pyfemtet_scripts.replace_model_considering_their_matching import ModelUpdater

