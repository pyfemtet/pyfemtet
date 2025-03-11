from __future__ import annotations

from v1.dask_util import *
from v1.logger import get_module_logger

__all__ = [
    'WorkerStatus',
    'ENTIRE_PROCESS_STATUS_KEY'
]

logger = get_module_logger('opt.worker_status', False)
ENTIRE_PROCESS_STATUS_KEY = 'entire_process_status'


class _WorkerStatus(float):

    string: str

    # noinspection PyUnusedLocal
    def __init__(self, x, string):
        self.string = string

    def __new__(cls, __x=..., string=...):
        return super().__new__(cls, __x)  # クラス変数インスタンス

    def __repr__(self):
        return self.string

    def str(self):
        return str(self)


class WorkerStatus:

    undefined = _WorkerStatus('nan', 'Undefined')
    initializing = _WorkerStatus('0', 'Initializing')
    launching_fem = _WorkerStatus(10, 'Launching FEM')
    waiting = _WorkerStatus(20, 'Waiting other workers')
    running = _WorkerStatus(30, 'Running')
    interrupting = _WorkerStatus(40, 'Interrupting')
    finished = _WorkerStatus(50, 'Finished')
    crashed = _WorkerStatus(60, 'Crashed')
    terminated = _WorkerStatus('inf', 'Terminated')

    @property
    def _dataset_name(self):
        if self.__dataset_name is None:
            worker = get_worker()
            if worker is not None:
                return worker.address + '_worker_status'
            else:
                return 'main_worker_status'
        else:
            return self.__dataset_name

    def __init__(self, dataset_name: str = None):
        self.__dataset_name = dataset_name
        self.__value: _WorkerStatus
        self.value = WorkerStatus.undefined

    @property
    def value(self) -> _WorkerStatus:
        client = get_client()
        if client is not None:
            key = self._dataset_name
            if key in client.list_datasets():
                return client.get_dataset(key)
            else:
                raise RuntimeError
        else:
            return self.__value

    @value.setter
    def value(self, value: _WorkerStatus):
        client = get_client()
        if client is not None:
            key = self._dataset_name
            if key in client.list_datasets():
                client.unpublish_dataset(key)
            client.publish_dataset(**{key: value})
        self.__value = value
