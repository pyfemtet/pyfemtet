from contextlib import nullcontext
from dask.distributed import Lock


def lock_or_no_lock(name: str, client=None):
    lock = Lock(name, client)
    if lock.client is None:
        return nullcontext()
    else:
        return lock
