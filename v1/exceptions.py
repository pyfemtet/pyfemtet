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
