import warnings
from packaging.version import Version
from threading import Lock as _ThreadingLock

import dask
from dask.distributed import LocalCluster, Client, Lock as _DaskLock, Nanny
from dask.distributed import get_client as _get_client, get_worker as _get_worker
from dask import config as cfg

from pyfemtet.logger import get_dask_logger, remove_all_output, get_module_logger

if Version(dask.__version__) < Version('2024.12.1'):
    import pyfemtet
    raise RuntimeError(
        f'pyfemtet {pyfemtet.__version__} requires dask >= 2024.12.1, '
        f'but the existing dask == {dask.__version__}. '
        f'Please consider to update dask.\n'
        f'ex: `py -m pip install dask distributed`'
    )

remove_all_output(get_dask_logger())

warnings.filterwarnings('ignore', category=RuntimeWarning, message="Couldn't detect a suitable IP address")

cfg.set({'distributed.scheduler.worker-ttl': None})
cfg.set({"distributed.scheduler.locks.lease-timeout": "inf"})

logger = get_module_logger('opt.dask', False)


__all__ = [
    'get_client',
    'get_worker',
    'Lock',
    'LocalCluster',
    'Client',
    'Nanny',
    'DummyClient',
]

_lock_pool = {}


def get_client(scheduler_address=None):
    try:
        return _get_client(scheduler_address)
    except ValueError:
        return None


def get_worker():
    try:
        return _get_worker()
    except ValueError:
        return None


def Lock(name, client=None):
    global _lock_pool

    if client is None:
        client = get_client()

    if client is not None:
        # import inspect
        _lock = _DaskLock(name)

    else:
        if name in _lock_pool:
            _lock = _lock_pool[name]
        else:
            _lock = _ThreadingLock()
            _lock_pool.update({name: _lock})

    return _lock


class DummyClient:

    @staticmethod
    def scheduler_info():
        return dict(workers=dict())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def map(self, *args, **kwargs): ...
    def gather(self, *args, **kwargs): ...
