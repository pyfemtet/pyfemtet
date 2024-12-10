from typing import List
from time import sleep
from threading import Thread

import pandas as pd

from pyfemtet.opt.visualization._base import PyFemtetApplicationBase, logger
from pyfemtet.opt.visualization._process_monitor.pages import HomePage, WorkerPage, PredictionModelPage, OptunaVisualizerPage
from pyfemtet._message import Msg


class ProcessMonitorApplication(PyFemtetApplicationBase):
    """"""
    """
        +------+--------+
        | side | con-   |
        | bar  | tent   |
        +--^---+--^-----+
           │      └─ pages (dict(href: str = layout: Component))
           └──────── nav_links (dict(order: float) = NavLink)

        Accessible members:
        - history: History
           └ local_df: pd.DataFrame
        - app: Dash
        - local_entire_status_int: int  <----------------> femopt.statue: OptimizationStatus(Actor)
        - local_worker_status_int_list: List[int]  <-----> femopt.opt.statue: List[OptimizationStatus(Actor)]
                                                     ^
                                                     |
                                   sync "while" statement in start_server()
    """

    DEFAULT_PORT = 8080

    def __init__(
            self,
            history,
            status,
            worker_addresses,
            worker_status_list,
            is_debug=False,
    ):
        super().__init__(
            title='PyFemtet Monitor',
            subtitle='visualize optimization progress',
            history=history,
        )

        self._should_get_actor_data = False
        self.is_debug = is_debug

        # type hint (avoid circular import)
        from pyfemtet.opt._femopt_core import OptimizationStatus

        # register arguments
        self.worker_addresses: List[str] = worker_addresses
        self.entire_status: OptimizationStatus = status  # include actor
        self.worker_status_list: List[OptimizationStatus] = worker_status_list  # include actor

        # initialize local members (from actors)
        self._df: pd.DataFrame = self.local_data
        self.local_entire_status_int: int = self.entire_status.get()
        self.local_worker_status_int_list: List[int] = [s.get() for s in self.worker_status_list]

    @property
    def local_data(self) -> pd.DataFrame:
        if self._should_get_actor_data:
            return self._df
        else:
            return self.history.get_df()

    @local_data.setter
    def local_data(self, value: pd.DataFrame):
        if self._should_get_actor_data:
            raise NotImplementedError('If should_get_actor_data, ProcessMonitorApplication.local_df is read_only.')
        else:
            self.history.set_df(value)

    def setup_callback(self, debug=False):
        if not debug:
            super().setup_callback()

    def start_server(self, host=None, port=None):
        """Callback の中で使いたい Actor のデータを Application クラスのメンバーとやり取りしつつ、server を落とす関数"""

        self._should_get_actor_data = True

        # avoid circular import
        from pyfemtet.opt._femopt_core import OptimizationStatus

        host = host or 'localhost'
        port = port or self.DEFAULT_PORT

        # dash app server を daemon thread で起動
        server_thread = Thread(
            target=self.run,
            args=(host, port,),
            daemon=True,
        )
        server_thread.start()

        # dash app (=flask server) の callback で dask の actor にアクセスすると
        # おかしくなることがあるので、ここで必要な情報のみやり取りする
        while True:
            # running 以前に monitor で current status を interrupting にしていれば actor に反映
            if (
                    (self.entire_status.get() <= OptimizationStatus.RUNNING)  # メインプロセスが RUNNING 以前である
                    and (self.local_entire_status_int == OptimizationStatus.INTERRUPTING)  # Application で status を INTERRUPT にした
            ):
                self.entire_status.set(OptimizationStatus.INTERRUPTING)
                for worker_status in self.worker_status_list:
                    worker_status.set(OptimizationStatus.INTERRUPTING)

            # status と df を actor から application に反映する
            self._df = self.history.get_df().copy()
            self.local_entire_status_int = self.entire_status.get()
            self.local_worker_status_int_list = [s.get() for s in self.worker_status_list]

            # terminate_all 指令があれば flask server をホストするプロセスごと終了する
            if self.entire_status.get() >= OptimizationStatus.TERMINATE_ALL:
                return 0  # take server down with me

            # interval
            sleep(1)

    @staticmethod
    def get_status_color(status_int):
        from pyfemtet.opt._femopt_core import OptimizationStatus
        # set color
        if status_int <= OptimizationStatus.SETTING_UP:
            color = 'secondary'
        elif status_int <= OptimizationStatus.WAIT_OTHER_WORKERS:
            color = 'primary'
        elif status_int <= OptimizationStatus.RUNNING:
            color = 'primary'
        elif status_int <= OptimizationStatus.INTERRUPTING:
            color = 'warning'
        elif status_int <= OptimizationStatus.TERMINATE_ALL:
            color = 'dark'
        elif status_int <= OptimizationStatus.CRASHED:
            color = 'danger'
        else:
            color = 'danger'
        return color


def g_debug():
    import os
    os.chdir(os.path.dirname(__file__))

    from pyfemtet.opt._femopt_core import History, OptimizationStatus

    class _OS(OptimizationStatus):

        # noinspection PyMissingConstructor
        def __init__(self, name):
            self.name = name
            self.st = self.INITIALIZING

        def set(self, status_const):
            self.st = status_const

        def get(self) -> int:
            return self.st

        def get_text(self) -> int:
            return self.const_to_str(self.st)

    g_application = ProcessMonitorApplication(
        history=History(history_path=os.path.join(os.path.dirname(__file__), '..', 'result_viewer', 'sample', 'sample.csv')),
        status=_OS('entire'),
        worker_addresses=['worker1', 'worker2', 'worker3'],
        worker_status_list=[_OS('worker1'), _OS('worker2'), _OS('worker3')],
        is_debug=False,
    )

    g_home_page = HomePage(Msg.PAGE_TITLE_PROGRESS)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_optuna = OptunaVisualizerPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/optuna', g_application)
    g_worker_page = WorkerPage(Msg.PAGE_TITLE_WORKERS, '/workers', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_optuna, 2)
    g_application.add_page(g_worker_page, 3)
    g_application.setup_callback(debug=False)

    g_application.run(debug=False)


def main(history, status, worker_addresses, worker_status_list, host=None, port=None):
    g_application = ProcessMonitorApplication(history, status, worker_addresses, worker_status_list)

    g_home_page = HomePage(Msg.PAGE_TITLE_PROGRESS)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_optuna = OptunaVisualizerPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/optuna', g_application)
    g_worker_page = WorkerPage(Msg.PAGE_TITLE_WORKERS, '/workers', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_optuna, 2)
    g_application.add_page(g_worker_page, 3)
    g_application.setup_callback()

    g_application.start_server(host, port)


if __name__ == '__main__':
    g_debug()
