import warnings

__all__ = [
    'ExceptionDuringOptimization',
    'FEMError',
    'HiddenConstraintViolation',
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


class HiddenConstraintViolation(ExceptionDuringOptimization):

    __subclasses__ = []

    @classmethod
    def __subclasshook__(cls, __subclass):
        cls.__subclasses__.append(__subclass)
        return super().__subclasshook__(__subclass)


class ModelError(FEMError, HiddenConstraintViolation): ...
class MeshError(FEMError, HiddenConstraintViolation): ...
class SolveError(FEMError, HiddenConstraintViolation): ...
class PostProcessError(FEMError, HiddenConstraintViolation): ...


class HardConstraintViolation(ExceptionDuringOptimization): ...
class InterruptOptimization(ExceptionDuringOptimization): ...
class SkipSolve(ExceptionDuringOptimization): ...


def show_experimental_warning(feature_name):
    warnings.warn(f'{feature_name} は実験的機能です。将来 API 等が'
                  f'大きく変更されるか、機能自体が削除される'
                  f'可能性があります。')
