from typing import List
from time import sleep
from threading import Thread

from flask import jsonify

# from pyfemtet.opt.visualization._base import PyFemtetApplicationBase, logger
# from pyfemtet.opt.visualization._process_monitor.pages import HomePage, WorkerPage, PredictionModelPage, OptunaVisualizerPage
# from pyfemtet._message import Msg

from v1.visualization2._base import PyFemtetApplicationBase
from v1.visualization2._process_monitor.pages import HomePage, WorkerPage, PredictionModelPage, OptunaVisualizerPage
from v1.worker_status import WorkerStatus
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

        self.is_debug = is_debug

        # type hint (avoid circular import)

        # register arguments
        self.local_data = history.get_df()  # scheduler への負荷を避けるためアクセスは while loop の中で行う
        self.worker_addresses: List[str] = worker_addresses
        self.entire_status: WorkerStatus = status  # include actor
        self.worker_status_list: List[WorkerStatus] = worker_status_list  # include actor

    def setup_callback(self, debug=False):
        if not debug:
            super().setup_callback()

        from pyfemtet.opt._femopt_core import OptimizationStatus

        @self.server.route('/interrupt')
        def some_command():

            # If the entire_state < INTERRUPTING, set INTERRUPTING
            if self.entire_status.value < WorkerStatus.interrupting:
                self.entire_status.value = WorkerStatus.interrupting
                result = {"message": "Interrupting signal emitted successfully."}

            else:
                result = {"message": "Interrupting signal is already emitted."}

            return jsonify(result)

    def start_server(self, host=None, port=None, host_record=None):
        """Callback の中で使いたい Actor のデータを Application クラスのメンバーとやり取りしつつ、server を落とす関数"""

        host = host or 'localhost'
        port = port or self.DEFAULT_PORT

        # dash app server を daemon thread で起動
        server_thread = Thread(
            target=self.run,
            args=(host, port,),
            kwargs=dict(host_record=host_record),
            daemon=True,
        )
        server_thread.start()

        while True:
            # df を actor から application に反映する
            self.local_data = self.history.get_df()

            # 一時的な実装

            # terminate_all 指令があれば flask server をホストするプロセスごと終了する
            if self.entire_status.value >= self.entire_status.terminated:
                return 0  # take server down with me

            # interval
            sleep(1)

    @staticmethod
    def get_status_color(status: WorkerStatus):
        # set color
        if status.value <= WorkerStatus.initializing:
            color = 'secondary'
        elif status.value <= WorkerStatus.waiting:
            color = 'primary'
        elif status.value <= WorkerStatus.running:
            color = 'primary'
        elif status.value <= WorkerStatus.interrupting:
            color = 'warning'
        elif status.value <= WorkerStatus.finished:
            color = 'dark'
        elif status.value <= WorkerStatus.crashed:
            color = 'danger'
        elif status.value <= WorkerStatus.terminated:
            color = 'dark'
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


def main(history, status, worker_addresses, worker_status_list, host=None, port=None, host_record=None):
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

    g_application.start_server(host, port, host_record)


if __name__ == '__main__':
    g_debug()
