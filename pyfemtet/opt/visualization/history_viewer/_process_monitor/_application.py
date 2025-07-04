from typing import List
from time import sleep
from threading import Thread

from flask import jsonify

from pyfemtet.logger import *
from pyfemtet._util.df_util import *

from pyfemtet.opt.worker_status import *
from pyfemtet.opt.visualization.history_viewer._base_application import *
from pyfemtet.opt.visualization.history_viewer._common_pages import *
from pyfemtet.opt.visualization.history_viewer._process_monitor._pages import *
from pyfemtet.opt.visualization.history_viewer._detail_page import DetailPage

from pyfemtet._i18n import Msg

logger = get_module_logger('opt.femopt', False)


class MonitorHostRecord:

    def __init__(self):
        self.info = dict()

    def set(self, host, port):
        self.info.update(dict(host=host, port=port))

    def get(self):
        return self.info


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
            worker_names,
            worker_status_list,
            is_debug=False,
    ):
        super().__init__(
            title='PyFemtet Monitor',
            subtitle='visualize optimization progress',
            history=history,
        )

        self.is_debug = is_debug

        # register arguments
        self._local_data = history.get_df()  # scheduler への負荷を避けるためアクセスは while loop の中で行う
        self.worker_names: List[str] = worker_names
        self.worker_addresses = worker_addresses
        self.entire_status: WorkerStatus = status  # include actor
        self.worker_status_list: List[WorkerStatus] = worker_status_list  # include actor

    def setup_callback(self, debug=False):
        if not debug:
            super().setup_callback()

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
            self._local_data = self.history.get_df()

            # terminate_all 指令があれば flask server をホストするプロセスごと終了する
            if self.entire_status.value >= self.entire_status.terminated:
                # monitor の worker status 更新を行う時間を待つ
                sleep(1)
                break

            # interval
            sleep(1)

        return 0  # take server down with me

    @staticmethod
    def get_status_color(status: WorkerStatus):
        # set color
        if status.value <= WorkerStatus.initializing:
            color = 'secondary'
        elif status.value <= WorkerStatus.launching_fem:
            color = 'info'
        elif status.value <= WorkerStatus.waiting:
            color = 'success'
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

    def get_df(self, equality_filters: dict = None):
        df = self._local_data

        if equality_filters is not None:
            df = get_partial_df(df, equality_filters)

        return df


def process_monitor_main(history, status, worker_addresses, worker_names, worker_status_list, host=None, port=None, host_record=None):
    g_application = ProcessMonitorApplication(history, status, worker_addresses, worker_names, worker_status_list)

    g_home_page = HomePage(Msg.PAGE_TITLE_PROGRESS, '/', g_application)
    g_rsm_page = PredictionModelPage(Msg.PAGE_TITLE_PREDICTION_MODEL, '/prediction-model', g_application)
    g_worker_page = WorkerPage(Msg.PAGE_TITLE_WORKERS, '/workers', g_application)
    g_detail = DetailPage(Msg.PAGE_TITLE_OPTUNA_VISUALIZATION, '/detail', g_application)

    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_rsm_page, 1)
    g_application.add_page(g_detail, 2)
    g_application.add_page(g_worker_page, 3)
    g_application.setup_callback()

    g_application.start_server(host, port, host_record)

    logger.info('Monitor terminated gracefully.')
