from time import sleep
from threading import Thread

from flask import Flask
from dash import Dash, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from dash import html, dcc

from v1.dask_util import *
from v1.history import *
from v1.worker_status import *
from v1.logger import get_module_logger, get_dash_logger, remove_all_output

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
            Output('label', 'children'),
            Input('interval', 'n_intervals'),
        )
        def update_graph(_):
            print('===== update_graph =====')

            client = get_client()
            if client is None:
                print('no client')
                raise PreventUpdate

            label = self.entire_status.value.str()

            return no_update, label


def run_monitor(
        history: History,
        entire_status: WorkerStatus,
        worker_status_list: list[WorkerStatus],
):
    """Dask process 上で terminate-able な Flask server を実行する関数"""

    client = get_client()
    assert client is not None

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
