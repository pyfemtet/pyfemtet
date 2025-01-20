import types
import importlib
from typing import TYPE_CHECKING, Any
from abc import ABCMeta


__all__ = [
    'isinstance_wrapper',
    '_LazyImport',
    '_lazy_class_factory',
]


class _MetaClass(ABCMeta):
    __original_cls__: type

    def __repr__(self):
        if hasattr(self, '__original_cls__'):
            return f'<LazyClass of {self.__original_cls__()}>'
        else:
            return f'<LazyClass of <unloaded class>>'


def _lazy_class_factory(module, cls_name):
    class LazyClass(object, metaclass=_MetaClass):
        # 継承を正しくコピーできていないので
        # issubclass を使う際は注意
        def __new__(cls, *args, **kwargs):
            OriginalClass = cls.__original_cls__()
            self = OriginalClass(*args, **kwargs)
            return self

        @staticmethod
        def __original_cls__():
            return getattr(module, cls_name)

    return LazyClass


class _LazyImport(types.ModuleType):
    """Module wrapper for lazy import.

    Note:
        This module is totally derived from `optuna._imports._LazyImport`.
        The following author has the copyright of this code.

        *******************************************
        Copyright (c) 2018 Preferred Networks, Inc.
        *******************************************

    This class wraps the specified modules and lazily imports them only when accessed.
    Otherwise, `import optuna` is slowed down by importing all submodules and
    dependencies even if not required.
    Within this project's usage, importlib override this module's attribute on the first
    access and the imported submodule is directly accessed from the second access.

    Args:
        name: Name of module to apply lazy import.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)


def isinstance_wrapper(obj: object, cls: type) -> bool:
    if isinstance(cls, _MetaClass):
        try:
            cls_ = cls.__original_cls__()
        except ModuleNotFoundError:
            return False
        return isinstance(obj, cls_)
    else:
        return isinstance(obj, cls)


if __name__ == '__main__':
    if TYPE_CHECKING:
        # for type check only
        import numpy
        NDArray = numpy.ndarray
        import no_module
    else:
        # runtime
        numpy = _LazyImport('numpy')
        NDArray = _lazy_class_factory(numpy, 'ndarray')
        NDArray2 = _lazy_class_factory(numpy, 'ndarray')
        no_module = _LazyImport('no_module')
        NoClass = _lazy_class_factory(no_module, 'NoClass')

    print('numpy is loaded:', numpy.__doc__ is not None)
    a = NDArray(shape=[1])
    print('numpy is loaded:', numpy.__doc__ is not None)

    print(f"{isinstance(None, NoClass)=}")
    print(f"{isinstance_wrapper(None, NoClass)=}")
    print(f"{isinstance(a, NDArray)=}")
    print(f"{isinstance_wrapper(a, NDArray)=}")
    print(f"{isinstance_wrapper(a, NoClass)=}")
    print(f"{isinstance_wrapper(a, NDArray2)=}")
