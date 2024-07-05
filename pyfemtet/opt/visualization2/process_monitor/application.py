# type hint
from dash.development.base_component import Component

# application
from dash import Dash
import webbrowser

# callback
from dash import Output, Input, State, no_update, callback_context
from dash.exceptions import PreventUpdate

# components
from dash import html, dcc
import dash_bootstrap_components as dbc

# graph
import pandas as pd
import plotly.express as px

# the others
from typing import List
from time import sleep
import logging
from threading import Thread
import psutil
from pyfemtet.logger import get_logger

from pyfemtet.opt.visualization2.base import PyFemtetApplicationBase


dash_logger = logging.getLogger('werkzeug')
logger = get_logger('viz')


class MonitorApplication(PyFemtetApplicationBase):

    DEFAULT_PORT = 8080

    def __init__(
            self,
            history,
            status,
            worker_addresses: List[str],
            worker_status_list: List["OptimizationStatus"],
    ):
        self.worker_addresses = worker_addresses
        self.entire_status = status  # include actor
        self.worker_status_list = worker_status_list  # include actor

        # start_server で更新するメンバーを一旦初期化
        self.local_df = history.local_data
        self.local_entire_status_int = self.entire_status.get()
        self.local_worker_status_int_list = [s.get() for s in self.worker_status_list]

        super().__init__(
            title='PyFemtet Monitor',
            subtitle='visualize optimization progress',
            history=history,
        )

    def start_server(
            self,
            host=None,
            port=None,
    ):
        """Callback の中で使いたい Actor のデータを Application クラスのメンバーとやり取りしつつ、server を落とす関数"""
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

            # status と df を actor から application に反映する
            self.local_entire_status_int = self.entire_status.get()
            self.local_worker_status_int_list = [s.get() for s in self.worker_status_list]
            self.local_df = self.history.actor_data.copy()

            # terminate_all 指令があれば flask server をホストするプロセスごと終了する
            if self.entire_status.get() >= OptimizationStatus.TERMINATE_ALL:
                return 0  # take server down with me

            # interval
            sleep(1)


