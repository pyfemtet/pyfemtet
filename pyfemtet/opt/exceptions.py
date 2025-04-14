import warnings

__all__ = [
    'ExceptionDuringOptimization',
    'FEMError',
    '_HiddenConstraintViolation',
    'ModelError',
    'MeshError',
    'SolveError',
    'PostProcessError',
    'HardConstraintViolation',
    'InterruptOptimization',
    'SkipSolve',
    'show_experimental_warning'
]


class ExceptionDuringOptimization(Exception): ...
class FEMError(Exception): ...


class _HiddenConstraintViolation(ExceptionDuringOptimization):

    __pyfemtet_subclasses__ = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _HiddenConstraintViolation.__pyfemtet_subclasses__.append(cls)


class ModelError(FEMError, _HiddenConstraintViolation): ...
class MeshError(FEMError, _HiddenConstraintViolation): ...
class SolveError(FEMError, _HiddenConstraintViolation): ...
class PostProcessError(FEMError, _HiddenConstraintViolation): ...


class HardConstraintViolation(ExceptionDuringOptimization): ...
class InterruptOptimization(ExceptionDuringOptimization): ...
class SkipSolve(ExceptionDuringOptimization): ...


def show_experimental_warning(feature_name):
    warnings.warn(f'{feature_name} は実験的機能です。将来 API 等が'
                  f'大きく変更されるか、機能自体が削除される'
                  f'可能性があります。')
