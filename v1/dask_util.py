import warnings
from contextlib import nullcontext

from dask.distributed import LocalCluster, Client, Lock as _Lock, Nanny
from dask.distributed import get_client as _get_client, get_worker as _get_worker
from dask import config as cfg

from v1.logger import get_dask_logger, remove_all_output

remove_all_output(get_dask_logger())

warnings.filterwarnings('ignore', category=RuntimeWarning, message="Couldn't detect a suitable IP address")

cfg.set({'distributed.scheduler.worker-ttl': None})


__all__ = [
    'get_client',
    'get_worker',
    'Lock',
    'LocalCluster',
    'Client',
    'Nanny'
]


def get_client():
    try:
        return _get_client()
    except ValueError:
        return None


def get_worker():
    try:
        return _get_worker()
    except ValueError:
        return None


def Lock(name, client=None):

    if client is None:
        client = get_client()

    if client:
        return _Lock(name, client)
    else:
        return nullcontext()
