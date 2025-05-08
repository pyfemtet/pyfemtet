import sys
from traceback import print_tb

__all__ = ['closing']


class closing:
    def __init__(self, thing):
        self.thing = thing

    def __enter__(self):
        return self.thing

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f'===== An exception is raised. Try to close resource. ===== ', file=sys.stderr)
            print_tb(exc_tb)
            print(f'{exc_type.__name__}: {exc_val}', file=sys.stderr)
        self.thing.close()
