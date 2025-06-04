from time import sleep
from pyfemtet._util.closing import closing as _closing


class closing(_closing):
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f'===== An exception is raised. Try to close resource. ===== ', file=sys.stderr)
            print_tb(exc_tb)
            print(f'{exc_type.__name__}: {exc_val}', file=sys.stderr)
        self.thing.close()
        sleep(5)
