import warnings


try:
    import brepmatching

except ModuleNotFoundError as e:
    warnings.warn(
        'There is no installation of `brepmatching`. '
        'Please confirm installation via '
        '`pip install pyfemtet[brep]` or '
        '`pip install brepmatching` command.'
    )
    raise e

except FileNotFoundError as e:
    warnings.warn(str(e))
    raise e
