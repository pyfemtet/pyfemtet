from __future__ import annotations

from time import sleep

from v1.utils.dask_util import *
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


def worker_status_from_float(value: float):
    if value == float('nan'):
        return _WorkerStatus(value, 'Undefined')
    elif value == float(0):
        return _WorkerStatus(value, 'Initializing')
    elif value == float(10):
        return _WorkerStatus(value, 'Launching FEM')
    elif value == float(20):
        return _WorkerStatus(value, 'Waiting for other workers')
    elif value == float(30):
        return _WorkerStatus(value, 'Running')
    elif value == float(40):
        return _WorkerStatus(value, 'Interrupting')
    elif value == float(50):
        return _WorkerStatus(value, 'Finished')
    elif value == float(60):
        return _WorkerStatus(value, 'Crashed')
    elif value == float('inf'):
        return _WorkerStatus(value, 'Terminated')


class WorkerStatus:

    undefined = worker_status_from_float(float('nan'))
    initializing = worker_status_from_float(0)
    launching_fem = worker_status_from_float(10)
    waiting = worker_status_from_float(20)
    running = worker_status_from_float(30)
    interrupting = worker_status_from_float(40)
    finished = worker_status_from_float(50)
    crashed = worker_status_from_float(60)
    terminated = worker_status_from_float(float('inf'))

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
            if client.scheduler is not None:
                key = self._dataset_name
                value: float | None = client.get_metadata(key, default=None)

                # value は単なる float になるので型変換
                if isinstance(value, float):
                    value: _WorkerStatus = worker_status_from_float(value)

                # setter の時点では client がなかった場合など
                elif value is None:
                    value = self.__value
                    client.set_metadata(key, value)

                return value

            # client はあるが、close された後である場合
            else:
                return self.__value
        else:
            return self.__value

    @value.setter
    def value(self, value: _WorkerStatus):
        client = get_client()
        if client is not None:
            if client.scheduler is not None:
                key = self._dataset_name
                client.set_metadata(key, value)
                # sleep(0.1)
        self.__value = value
