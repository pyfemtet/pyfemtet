import os
import dataclasses
from contextlib import nullcontext
import datetime
from time import sleep
import enum
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from typing import (
    Callable, Any, SupportsFloat,
    Sequence,
)

import pandas as pd

try:
    # noinspection PyUnresolvedReferences
    from pythoncom import CoInitialize, CoUninitialize
    from win32com.client import Dispatch, Constants, constants
except ModuleNotFoundError:
    CoInitialize = lambda: None
    CoUninitialize = lambda: None
    Dispatch = type('NoDispatch', (object,), {})
    Constants = type('NoConstants', (object,), {})
    constants = Constants()

from dask.distributed import LocalCluster, Client, Lock, Nanny
from dask.distributed import get_client as _get_client, get_worker as _get_worker
from dask import config as cfg

from flask import Flask
from dash import Dash, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from dash import html, dcc

cfg.set({'distributed.scheduler.worker-ttl': None})


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


TrialInput = dict[str, SupportsFloat]
TrialOutput = dict[str, SupportsFloat]
Fidelity = SupportsFloat | str | None


class DataFrameWrapper:
    __df: pd.DataFrame
    _lock_name = 'edit-df'
    _dataset_name = 'df'

    def __init__(self, df: pd.DataFrame):
        self.set_df(df)

    def __len__(self):
        return len(self.get_df())

    def __str__(self):
        return self.get_df().__str__()

    @property
    def lock(self):
        client = get_client()
        if client:
            return Lock(self._lock_name)
        else:
            return nullcontext()

    def get_df(self):
        client = get_client()
        if client:
            if self._dataset_name in client.list_datasets():
                return client.get_dataset(self._dataset_name)
            else:
                raise RuntimeError
        else:
            return self.__df

    def set_df(self, df):
        client = get_client()
        if client:
            if self._dataset_name in client.list_datasets():
                client.unpublish_dataset(self._dataset_name)
            client.publish_dataset(**dict(df=df))
        self.__df = df

    def start_dask(self):
        # Register the df initialized before dask context.
        self.set_df(self.__df)

    def end_dask(self):
        # Get back the df on dask to use the value outside
        # dask context.
        self.__df = self.get_df()


class TrialState(enum.Enum):
    succeeded = 'Success'
    hard_constraint_violation = 'Hard constraint violation'
    soft_constraint_violation = 'Soft constraint violation'
    model_error = 'Model error'
    mesh_error = 'Mesh error'
    solve_error = 'Solve error'
    post_error = 'Post-processing error'
    unknown_error = 'Unknown error'


Record = NotImplemented


class Records:
    """最適化の試行全体の情報を格納するモデルクラス"""
    df_wrapper: DataFrameWrapper

    def __init__(self):
        self.df_wrapper = DataFrameWrapper(pd.DataFrame())

    def __str__(self):
        return self.df_wrapper.__str__()

    def __len__(self):
        return len(self.df_wrapper)

    @property
    def lock(self):
        return self.df_wrapper.lock


class History:
    """最適化の試行全体の情報を操作するルールクラス"""
    _records: Records

    def __init__(self):
        self._records = Records()
        self.path: str
        self.current_trial_time_start: datetime.datetime = None

    def __str__(self):
        return self._records.__str__()

    def __enter__(self):
        self._records.df_wrapper.start_dask()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._records.df_wrapper.end_dask()

    def _trial_start(self):
        self.current_trial_time_start = datetime.datetime.now()

    def _trial_end(self):
        self.current_trial_time_start = None

    def trial_recording(self):

        class TrialContext:

            # noinspection PyMethodParameters
            def __enter__(_self):
                self._trial_start()

            # noinspection PyMethodParameters
            def __exit__(_self, exc_type, exc_val, exc_tb):
                self._trial_end()

        return TrialContext()

    def get_df(self):
        return self._records.df_wrapper.get_df()


class Monitor:

    def __init__(self, history: History):
        self.server = Flask(__name__)
        self.app = Dash(server=self.server)
        self.history = history

        self.setup_layout()
        self.setup_callback()

    def setup_layout(self):
        self.app.layout = html.Div(
            children=[
                dcc.Interval(interval=1000, id='interval'),
                html.Button(children='refresh', id='button'),
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

            client = get_client()
            if client is None:
                print('no client')
                raise PreventUpdate

            print(self.history)

            return no_update


def run_monitor(history: History):
    """Dask process 上で terminate-able な Flask server を実行する関数"""

    client = get_client()
    assert client is not None

    monitor = Monitor(history)
    t = Thread(
        target=monitor.app.run,
        kwargs=dict(debug=False),
        daemon=True,
    )
    t.start()

    while client.get_dataset('status', default='running') == 'running':
        sleep(1)

    print('Monitor terminated gracefully.')


class AbstractFEMInterface:
    def update(self, x: 'TrialInput'):
        raise NotImplementedError

    def _get_worker_space(self):
        worker = get_worker()
        if worker is None:
            return os.getcwd()
        else:
            return worker.local_directory

    def _distribute_files(self, paths: list[str]):

        client = get_client()
        if client is None:
            return

        for path in paths:

            if not os.path.exists(path):
                raise FileNotFoundError

            client.upload_file(path, load=False)

    def _setup_after_parallel(self):
        pass


class FemtetInterface(AbstractFEMInterface):
    _com_members = {'Femtet': 'FemtetMacro.Femtet'}

    def __init__(self):
        self.Femtet = Dispatch('FemtetMacro.Femtet')

    def __getstate__(self):
        """Pickle するメンバーから COM を除外する"""
        state = self.__dict__.copy()
        for key in self._com_members.keys():
            del state[key]
        return state

    def __setstate__(self, state):
        """UnPickle 時に COM を再構築する"""
        CoInitialize()
        for key, value in state['_com_members'].items():
            print(os.getpid())
            state.update({key: Dispatch(value)})
        self.__dict__.update(state)

    def _setup_after_parallel(self):
        CoInitialize()  # Main Process sub thread では __setstate__ が呼ばれない


class NoFEM(AbstractFEMInterface):
    ...


class Function:
    _fun: Callable[..., SupportsFloat]
    args: tuple
    kwargs: dict

    @property
    def fun(self) -> Callable[..., SupportsFloat]:
        self._ScapeGoat.restore_constants(self._fun)
        return self._fun

    @fun.setter
    def fun(self, f: Callable[..., SupportsFloat]):
        self._fun = f

    def __getstate__(self):
        """Pickle 時に _fun が参照する constants を _ScapeGoat にする"""
        state = self.__dict__
        if '_fun' in state:
            self._ScapeGoat.remove_constants(state['_fun'])
        return state

    def __setstate__(self, state):
        """Pickle 時に _fun が参照する _ScapeGoat を constants にする"""
        CoInitialize()
        if '_fun' in state:
            self._ScapeGoat.restore_constants(state['_fun'])
        self.__dict__.update(state)

    class _ScapeGoat:

        @classmethod
        def restore_constants(cls, f: ...):
            """f の存在する global スコープの _Scapegoat 変数を constants に変更"""
            if not hasattr(f, '__globals__'):
                return

            for name, var in f.__globals__.items():
                if isinstance(var, cls):
                    # try 不要
                    # fun の定義がこのファイル上にある場合、つまりデバッグ時のみ
                    # remove_constants がこのスコープの constants を消すので
                    # constants を再インポートする必要がある
                    from win32com.client import constants
                    f.__globals__[name] = constants

        @classmethod
        def remove_constants(cls, f: ...):
            """f の存在する global スコープの Constants 変数を _Scapegoat に変更"""

            if not hasattr(f, '__globals__'):
                return

            for name, var in f.__globals__.items():
                if isinstance(var, Constants):
                    f.__globals__[name] = cls()

    def eval(self) -> SupportsFloat:
        return self.fun(*self.args, **self.kwargs)


class Functions(dict[str, Function]): ...


class Objective(Function):
    direction: str | SupportsFloat

    def eval(self) -> tuple[SupportsFloat, SupportsFloat]:
        value = Function.eval(self)

        if self.direction == 'maximize':
            value_as_minimize = value
        elif self.direction == 'maximize':
            value_as_minimize = -value
        elif isinstance(self.direction, SupportsFloat):
            value_as_minimize = (value - self.direction) ** 2
        else:
            raise NotImplementedError

        return value, value_as_minimize


class Objectives(dict[str, Objective]): ...


class Variable:

    value: SupportsFloat

    def __init__(self, name: str):
        self.name = name


class Parameter(Variable):

    lower_bound: SupportsFloat
    upper_bound: SupportsFloat



class VariableManager(dict[str, Variable]):
    ...


class AbstractOptimizer:

    variable_manager: VariableManager
    objectives: Objectives

    history: History
    fem: AbstractFEMInterface
    current_trial_start: datetime.datetime | None

    def __init__(self):

        # Problem
        self.variable_manager = VariableManager()
        self.objectives = {}

        # System
        self.fem = None
        self.history = History()

        # Util
        self.current_trial_start = None

    def add_parameter(
            self,
            name: str,
            initial_value: SupportsFloat | None,
            lower_bound: SupportsFloat | None,
            upper_bound: SupportsFloat | None,
    ) -> None:
        parameter = Parameter(name)
        parameter.value = initial_value
        parameter.lower_bound = lower_bound
        parameter.upper_bound = upper_bound
        self.variable_manager.update({name: parameter})

    def add_objective(
            self,
            name: str,
            fun: Callable[[...], SupportsFloat],
            direction: str | SupportsFloat = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> None:
        obj = Objective()
        obj.fun = fun
        obj.args = args or ()
        obj.kwargs = kwargs or {}
        obj.direction = direction
        self.objectives.update({name: obj})

    def f(self, x: 'TrialInput') -> 'TrialOutput':

        # Update FEM
        self.fem.update(x)

        # Returns output
        out = TrialOutput()
        for name, obj in self.objectives.items():
            obj_value, obj_value_internal = obj.eval()
            out.update({name: obj_value_internal})

        return out

    def register(self, x: 'TrialInput', y: 'TrialOutput') -> None:
        self.history.register(
            x,
            y,
        )

    def run(self) -> None:

        # loop this
        for i in range(5):
            with self.history.record():
                self.history.trial_start()
                next_input = self.suggest()
                output = self.probe(next_input)
                self.register(next_input, output)
            sleep(3)

    def _run(self, worker_idx) -> None:
        print(f'worker {worker_idx}')

        self.fem._setup_after_parallel()

        self.run()
        print(f'worker {worker_idx} complete!')


class FEMOpt:
    opt: AbstractOptimizer
    local = True

    def optimize(self, n_parallel) -> None:
        _cluster = LocalCluster(
            n_workers=n_parallel - 1 + 1,
            threads_per_worker=1,
            processes=True,
        )
        _client = Client(
            _cluster
        )
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='thread_worker'
        )

        with _cluster, _client as client, self.opt.history:

            # Get workers
            nannies: tuple[Nanny] = tuple(client.cluster.workers.values())

            # Assign roles
            monitor_worker_address = nannies[0].worker_address
            opt_worker_addresses = [n.worker_address for n in nannies[1:]]

            # Setting up monitor
            monitor_future = client.submit(
                run_monitor,
                # Arguments of func
                history=self.opt.history,
                # Arguments of submit
                workers=(monitor_worker_address,),
                allow_other_workers=False,
            )

            # Run on main process
            future = executor.submit(
                self.opt._run,
                'Main'
            )

            # Run on cluster
            futures = client.map(
                self.opt._run,
                # Arguments of func
                range(n_parallel - 1),
                # Arguments of map
                workers=opt_worker_addresses,
                allow_other_workers=False,
            )

            # Wait to finish optimization
            client.gather(futures)
            future.result()

            # Send termination signal to monitor
            # and wait to finish
            # noinspection PyTypeChecker
            client.publish_dataset(status='terminated')
            monitor_future.result()


def _obj():
    print(os.getpid())
    print(f'{constants.STATIC_C=}')
    return type(constants)


if __name__ == '__main__':
    objective = Objective()
    objective.fun = _obj

    objectives = Objectives()
    objectives['sample'] = objective

    opt = AbstractOptimizer()
    opt.objectives = objectives
    opt.fem = NoFEM()

    femopt = FEMOpt()
    femopt.opt = opt

    femopt.optimize(n_parallel=3)

    print(femopt.opt.history)
