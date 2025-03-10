__all__ = [
    'FEMError',
    'HiddenConstraintViolation',
    'ModelError',
    'MeshError',
    'SolveError',
    'PostProcessError',
]


class HiddenConstraintViolation(Exception):

    __subclasses__ = []

    @classmethod
    def __subclasshook__(cls, __subclass):
        cls.__subclasses__.append(__subclass)
        return super().__subclasshook__(__subclass)


class FEMError(Exception): ...
class ModelError(FEMError, HiddenConstraintViolation): ...
class MeshError(FEMError, HiddenConstraintViolation): ...
class SolveError(FEMError, HiddenConstraintViolation): ...
class PostProcessError(FEMError, HiddenConstraintViolation): ...
