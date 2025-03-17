from time import sleep
from threading import Thread

from flask import Flask
from dash import Dash, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from dash import html, dcc

from v1.utils.dask_util import *
from v1.history import *
from v1.worker_status import *
from v1.logger import get_module_logger, get_dash_logger, remove_all_output

from v1.prediction.model import *
from v1.monitor.plotter.plot_prediction_model import plot3d

logger = get_module_logger('opt.femopt')
remove_all_output(get_dash_logger())


__all__ = ['run_monitor']


class Monitor:

    def __init__(
            self,
            history: History,
            entire_status: WorkerStatus,
            worker_status_list: list[WorkerStatus],
    ):
        self.server = Flask(__name__)
        self.app = Dash(server=self.server)
        self.history = history
        self.entire_status = entire_status
        self.worker_status_list = worker_status_list

        self.pyfemtet_model: PyFemtetModel = PyFemtetModel()

        self.setup_layout()
        self.setup_callback()



    def setup_layout(self):
        self.app.layout = html.Div(
            children=[
                dcc.Interval(interval=500, id='interval'),
                html.Button(children='refresh', id='button'),
                html.Label(children='Status Here', id='label'),
                dcc.Graph(id='graph'),
            ]
        )

    def setup_callback(self):
        @self.app.callback(
            Output('graph', 'figure'),
            Input('interval', 'n_intervals'),
        )
        def update_graph(_):
            print('===== update_graph =====')

            # label = self.entire_status.value.str()

            # set default value
            equality_filters = History.MAIN_FILTER

            # get df to create figure
            df = self.history.get_df(
                equality_filters=equality_filters
            )

            model = SingleTaskGPModel()
            self.pyfemtet_model.update_model(model)
            self.pyfemtet_model.fit(
                self.history,
                df,
                **{}
            )

            fig = plot3d(
                self.history,
                'x1',
                'x2',
                {},
                'obj1',
                df,
                self.pyfemtet_model,
                20,
            )

            if fig is None:
                logger.warning('まだグラフが描けるほど結果がありません。')
                return no_update

            return fig


def run_monitor(
        history: History,
        entire_status: WorkerStatus,
        worker_status_list: list[WorkerStatus],
):
    """Dask process 上で terminate-able な Flask server を実行する関数"""

    monitor = Monitor(history, entire_status, worker_status_list)
    t = Thread(
        target=monitor.app.run,
        kwargs=dict(debug=False),
        daemon=True,
    )
    t.start()

    while entire_status.value < WorkerStatus.terminated:
        sleep(1)

    logger.info('Monitor terminated gracefully.')
