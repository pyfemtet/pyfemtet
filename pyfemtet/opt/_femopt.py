# built-in
import os
import datetime
from time import time, sleep
from threading import Thread
import json

# 3rd-party
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster, Client

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface, FemtetInterface
from pyfemtet.opt.opt import AbstractOptimizer, OptunaOptimizer
from pyfemtet.opt.visualization.process_monitor.application import main as process_monitor_main
from pyfemtet.opt._femopt_core import (
    _check_bound,
    _is_access_gogh,
    Objective,
    Constraint,
    History,
    OptimizationStatus,
    logger,
)
from pyfemtet.message import Msg, encoding


class FEMOpt:
    """Class to control FEM interface and optimizer.

    Args:
        fem (FEMInterface, optional): The finite element method interface. Defaults to None. If None, automatically set to FemtetInterface.
        opt (AbstractOptimizer):
        history_path (str, optional): The path to the history file. Defaults to None. If None, '%Y_%m_%d_%H_%M_%S.csv' is created in current directory.
        scheduler_address (str or None): If cluster processing, set this parameter like "tcp://xxx.xxx.xxx.xxx:xxxx".

    Attributes:
        fem (FEMInterface): The interface of FEM system.
        opt (AbstractOptimizer): The optimizer.
        scheduler_address (str or None): Dask scheduler address. If None, LocalCluster will be used.
        client (Client): Dask client. For detail, see dask documentation.
        status (OptimizationStatus): Entire process status. This contains dask actor.
        history(History): History of optimization process. This contains dask actor.
        history_path (str): The path to the history (.csv) file.
        worker_status_list([OptimizationStatus]): Process status of each dask worker.
        monitor_process_future(Future): Future of monitor server process. This is dask future.
        monitor_server_kwargs(dict): Monitor server parameter. Currently, the valid arguments are hostname and port.

    """

    def __init__(
            self,
            fem: FEMInterface = None,
            opt: AbstractOptimizer = None,
            history_path: str = None,
            scheduler_address: str = None
    ):
        logger.info('Initialize FEMOpt')

        # 引数の処理
        if history_path is None:
            history_path = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.csv')
        self.history_path = os.path.abspath(history_path)
        self.scheduler_address = scheduler_address

        if fem is None:
            self.fem = FemtetInterface()
        else:
            self.fem = fem

        if opt is None:
            self.opt = OptunaOptimizer()
        else:
            self.opt = opt

        # メンバーの宣言
        self.client = None
        self.status = None  # actor
        self.history = None  # actor
        self.worker_status_list = None  # [actor]
        self.monitor_process_future = None
        self.monitor_server_kwargs = dict()
        self.monitor_process_worker_name = None
        self._is_error_exit = False

    # multiprocess 時に pickle できないオブジェクト参照の削除
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fem']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_random_seed(self, seed: int):
        """Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed value to be set.

        """
        self.opt.seed = seed

    def add_parameter(
            self,
            name: str,
            initial_value: float or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            step: float or None = None,
            memo: str = ''
    ):
        """Adds a parameter to the optimization problem.

        Args:
            name (str): The name of the parameter.
            initial_value (float or None, optional): The initial value of the parameter. Defaults to None. If None, try to get initial value from FEMInterface.
            lower_bound (float or None, optional): The lower bound of the parameter. Defaults to None. However, this argument is required for some algorithms.
            upper_bound (float or None, optional): The upper bound of the parameter. Defaults to None. However, this argument is required for some algorithms.
            step (float or None, optional): The step of parameter. Defaults to None.
            memo (str, optional): Additional information about the parameter. Defaults to ''.
        Raises:
            ValueError: If initial_value is not specified and the value for the given name is also not specified.

        """

        _check_bound(lower_bound, upper_bound, name)
        value = self.fem.check_param_value(name)
        if initial_value is None:
            if value is not None:
                initial_value = value
            else:
                raise ValueError('initial_value を指定してください.')

        d = {
            'name': name,
            'value': float(initial_value),
            'lb': float(lower_bound) if lower_bound is not None else None,
            'ub': float(upper_bound) if upper_bound is not None else None,
            'step': float(step) if step is not None else None,
            'memo': memo,
        }
        pdf = pd.DataFrame(d, index=[0], dtype=object)

        if len(self.opt.parameters) == 0:
            self.opt.parameters = pdf
        else:
            self.opt.parameters = pd.concat([self.opt.parameters, pdf], ignore_index=True)

    def add_objective(
            self,
            fun,
            name: str or None = None,
            direction: str or float = 'minimize',
            args: tuple or None = None,
            kwargs: dict or None = None
    ):
        """Adds an objective to the optimization problem.

        Args:
            fun (callable): The objective function.
            name (str or None, optional): The name of the objective. Defaults to None.
            direction (str or float, optional): The optimization direction. Defaults to 'minimize'.
            args (tuple or None, optional): Additional arguments for the objective function. Defaults to None.
            kwargs (dict or None, optional): Additional keyword arguments for the objective function. Defaults to None.

        Note:
            If the FEMInterface is FemtetInterface, the 1st argument of fun should be Femtet (IPyDispatch) object.

        Tip:
            If name is None, name is a string with the prefix `"obj_"` followed by a sequential number.

        """

        # 引数の処理
        if args is None:
            args = tuple()
        elif not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        if name is None:
            prefix = Objective.default_name
            i = 0
            while True:
                candidate = f'{prefix}_{str(int(i))}'
                is_existing = candidate in list(self.opt.objectives.keys())
                if not is_existing:
                    break
                else:
                    i += 1
            name = candidate

        self.opt.objectives[name] = Objective(fun, name, direction, args, kwargs)

    def add_constraint(
            self,
            fun,
            name: str or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            strict: bool = True,
            args: tuple or None = None,
            kwargs: dict or None = None,
    ):
        """Adds a constraint to the optimization problem.

        Args:
            fun (callable): The constraint function.
            name (str or None, optional): The name of the constraint. Defaults to None.
            lower_bound (float or Non, optional): The lower bound of the constraint. Defaults to None.
            upper_bound (float or Non, optional): The upper bound of the constraint. Defaults to None.
            strict (bool, optional): Flag indicating if it is a strict constraint. Defaults to True.
            args (tuple or None, optional): Additional arguments for the constraint function. Defaults to None.
            kwargs (dict): Additional arguments for the constraint function. Defaults to None.

        Note:
            If the FEMInterface is FemtetInterface, the 1st argument of fun should be Femtet (IPyDispatch) object.

        Tip:
            If name is None, name is a string with the prefix `"cns_"` followed by a sequential number.

        """

        # 引数の処理
        if args is None:
            args = tuple()
        elif not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        if name is None:
            prefix = Constraint.default_name
            i = 0
            while True:
                candidate = f'{prefix}_{str(int(i))}'
                is_existing = candidate in list(self.opt.constraints.keys())
                if not is_existing:
                    break
                else:
                    i += 1
            name = candidate

        # strict constraint の場合、solve 前に評価したいので Gogh へのアクセスを禁ずる
        if strict:
            if _is_access_gogh(fun):
                message = Msg.ERR_CONTAIN_GOGH_ACCESS_IN_STRICT_CONSTRAINT
                raise Exception(message)

        self.opt.constraints[name] = Constraint(fun, name, lower_bound, upper_bound, strict, args, kwargs)

    def get_parameter(self, format='dict') -> pd.DataFrame or dict or np.ndarray:
        """Returns the parameter in a specified format.

        Args:
            format (str, optional): The desired output format. Defaults to 'dict'. Valid formats are 'values', 'df' and 'dict'.

        Returns:
            pd.DataFrame or dict or np.ndarray: The parameter data converted into the specified format.

        Raises:
            ValueError: If an invalid format is provided.

        """
        return self.opt.get_parameter(format)

    def set_monitor_host(self, host=None, port=None):
        """Sets up the monitor server with the specified host and port.

        Args:
            host (str): The hostname or IP address of the monitor server.
            port (int or None, optional): The port number of the monitor server. If None, ``8080`` will be used. Defaults to None.

        Tip:
            Specifying host ``0.0.0.0`` allows viewing monitor from all computers on the local network.

            However, please note that in this case,
            it will be visible to all users on the local network.

            If no hostname is specified, the monitor server will be hosted on ``localhost``.

        """
        self.monitor_server_kwargs = dict(
            host=host,
            port=port
        )

    def optimize(
            self,
            n_trials=None,
            n_parallel=1,
            timeout=None,
            wait_setup=True,
    ):
        """Runs the main optimization process.

        Args:
            n_trials (int or None, optional): The number of trials. Defaults to None.
            n_parallel (int, optional): The number of parallel processes. Defaults to 1.
            timeout (float or None, optional): The maximum amount of time in seconds that each trial can run. Defaults to None.
            wait_setup (bool, optional): Wait for all workers launching FEM system. Defaults to True.

        Tip:
            If set_monitor_host() is not executed, a local server for monitoring will be started at localhost:8080.

        Note:
            If ``n_trials`` and ``timeout`` are both None, it runs forever until interrupting by the user.

        Note:
            If ``n_parallel`` >= 2, depending on the end timing, ``n_trials`` may be exceeded by up to ``n_parallel-1`` times.

        Warning:
            If ``n_parallel`` >= 2 and ``fem`` is a subclass of ``FemtetInterface``, the ``strictly_pid_specify`` of subprocess is set to ``False``.
            So **it is recommended to close all other Femtet processes before running.**

        """

        # method checker
        if n_parallel > 1:
            self.opt.method_checker.check_parallel()

        if timeout is not None:
            self.opt.method_checker.check_timeout()

        if len(self.opt.objectives) > 1:
            self.opt.method_checker.check_multi_objective()

        if len(self.opt.constraints) > 0:
            self.opt.method_checker.check_constraint()

        for key, value in self.opt.constraints.items():
            if value.strict:
                self.opt.method_checker.check_strict_constraint()
                break

        if self.opt.seed is not None:
            self.opt.method_checker.check_seed()

        is_incomplete_bounds = False
        for _, row in self.opt.parameters.iterrows():
            lb, ub = row['lb'], row['ub']
            is_incomplete_bounds = is_incomplete_bounds + (lb is None) + (ub is None)
        if is_incomplete_bounds:
            self.opt.method_checker.check_incomplete_bounds()

        # 共通引数
        self.opt.n_trials = n_trials
        self.opt.timeout = timeout

        # クラスターの設定
        self.opt.is_cluster = self.scheduler_address is not None
        if self.opt.is_cluster:
            # 既存のクラスターに接続
            logger.info('Connecting to existing cluster.')
            self.client = Client(self.scheduler_address)

            # 最適化タスクを振り分ける worker を指定
            subprocess_indices = list(range(n_parallel))
            worker_addresses = list(self.client.nthreads().keys())

            # worker が足りない場合はエラー
            if n_parallel > len(worker_addresses):
                raise RuntimeError(f'n_parallel({n_parallel}) > n_workers({len(worker_addresses)}). There are insufficient number of workers.')

            # worker が多い場合は閉じる
            if n_parallel < len(worker_addresses):
                used_worker_addresses = worker_addresses[:n_parallel]  # 前から順番に選ぶ：CPU の早い / メモリの多い順に並べることが望ましい
                unused_worker_addresses = worker_addresses[n_parallel:]
                self.client.retire_workers(unused_worker_addresses, close_workers=True)
                worker_addresses = used_worker_addresses

            # monitor worker の設定
            logger.info('Launching monitor server. This may take a few seconds.')
            self.monitor_process_worker_name = datetime.datetime.now().strftime("Monitor%Y%m%d%H%M%S")
            current_n_workers = len(self.client.nthreads().keys())
            from subprocess import Popen
            import sys
            Popen(
                f'{sys.executable} -m dask worker {self.client.scheduler.address} --nthreads 1 --nworkers 1 --name {self.monitor_process_worker_name} --no-nanny',
                shell=True
            )

            # monitor 用 worker が増えるまで待つ
            self.client.wait_for_workers(n_workers=current_n_workers + 1)

        else:
            # ローカルクラスターを構築
            logger.info('Launching single machine cluster. This may take tens of seconds.')
            cluster = LocalCluster(processes=True, n_workers=n_parallel,
                                   threads_per_worker=1)  # n_parallel = n_parallel - 1 + 1; main 分減らし、monitor 分増やす
            self.client = Client(cluster, direct_to_workers=False)
            self.scheduler_address = self.client.scheduler.address

            # 最適化タスクを振り分ける worker を指定
            subprocess_indices = list(range(n_parallel))[1:]
            worker_addresses = list(self.client.nthreads().keys())

            # monitor worker の設定
            self.monitor_process_worker_name = worker_addresses[0]
            worker_addresses[0] = 'Main'

        # Femtet 特有の処理
        metadata = None
        if isinstance(self.fem, FemtetInterface):
            # 結果 csv に記載する femprj に関する情報
            metadata = json.dumps(
                dict(
                    femprj_path=self.fem.original_femprj_path,
                    model_name=self.fem.model_name
                )
            )
            # Femtet の parametric 設定を目的関数に用いるかどうか
            if self.fem.parametric_output_indexes_use_as_objective is not None:
                from pyfemtet.opt.interface._femtet_parametric import add_parametric_results_as_objectives
                indexes = list(self.fem.parametric_output_indexes_use_as_objective.keys())
                directions = list(self.fem.parametric_output_indexes_use_as_objective.values())
                add_parametric_results_as_objectives(
                    self,
                    indexes,
                    directions,
                )

        # actor の設定
        self.status = OptimizationStatus(self.client)
        self.worker_status_list = [OptimizationStatus(self.client, name) for name in worker_addresses]  # tqdm 検討
        self.status.set(OptimizationStatus.SETTING_UP)
        self.history = History(
            self.history_path,
            self.opt.parameters['name'].to_list(),
            list(self.opt.objectives.keys()),
            list(self.opt.constraints.keys()),
            self.client,
            metadata,
        )

        # launch monitor
        self.monitor_process_future = self.client.submit(
            # func
            _start_monitor_server,
            # args
            self.history,
            self.status,
            worker_addresses,
            self.worker_status_list,
            # kwargs
            **self.monitor_server_kwargs,
            # kwargs of submit
            workers=self.monitor_process_worker_name,
            allow_other_workers=False
        )

        # fem
        self.fem._setup_before_parallel(self.client)

        # opt
        self.opt.fem_class = type(self.fem)
        self.opt.fem_kwargs = self.fem.kwargs
        self.opt.entire_status = self.status
        self.opt.history = self.history
        self.opt._setup_before_parallel()

        # クラスターでの計算開始
        self.status.set(OptimizationStatus.LAUNCHING_FEM)
        start = time()
        calc_futures = self.client.map(
            self.opt._run,
            subprocess_indices,
            [self.worker_status_list] * len(subprocess_indices),
            [wait_setup] * len(subprocess_indices),
            workers=worker_addresses,
            allow_other_workers=False,
        )

        t_main = None
        if not self.opt.is_cluster:
            # ローカルプロセスでの計算(opt._main 相当の処理)
            subprocess_idx = 0

            # set_fem
            self.opt.fem = self.fem
            self.opt._reconstruct_fem(skip_reconstruct=True)

            t_main = Thread(
                target=self.opt._run,
                args=(
                    subprocess_idx,
                    self.worker_status_list,
                    wait_setup,
                ),
                kwargs=dict(
                    skip_set_fem=True,
                )
            )
            t_main.start()

        # save history
        def save_history():
            while True:
                sleep(2)
                try:
                    self.history.save()
                except PermissionError:
                    logger.warning(Msg.WARN_HISTORY_CSV_NOT_ACCESSIBLE)
                if self.status.get() >= OptimizationStatus.TERMINATED:
                    break

        t_save_history = Thread(target=save_history)
        t_save_history.start()

        # 終了を待つ
        local_opt_crashed = False
        opt_crashed_list = self.client.gather(calc_futures)
        if not self.opt.is_cluster:  # 既存の fem を使っているならそれも待つ
            if t_main is not None:
                t_main.join()
                local_opt_crashed = self.opt._is_error_exit
        opt_crashed_list.append(local_opt_crashed)
        self.status.set(OptimizationStatus.TERMINATED)
        end = time()

        # 一応
        t_save_history.join()

        # logger.info(f'計算が終了しました. 実行時間は {int(end - start)} 秒でした。ウィンドウを閉じると終了します.')
        # logger.info(f'結果は{self.history.path}を確認してください.')
        logger.info(Msg.OPTIMIZATION_FINISHED)
        logger.info(self.history.path)

        # ひとつでも crashed ならばフラグを立てる
        if any(opt_crashed_list):
            self._is_error_exit = True
        
        return self.history.local_data


    def terminate_all(self):
        """Try to terminate all launched processes.

        If distributed computing, Scheduler and Workers will NOT be terminated.

        """

        # monitor が terminated 状態で少なくとも一度更新されなければ running のまま固まる
        sleep(1)

        # terminate monitor process
        self.status.set(OptimizationStatus.TERMINATE_ALL)
        logger.info(self.monitor_process_future.result())
        sleep(1)

        # terminate actors
        self.client.cancel(self.history._future, force=True)
        self.client.cancel(self.status._future, force=True)
        for worker_status in self.worker_status_list:
            self.client.cancel(worker_status._future, force=True)
        logger.info('Terminate actors.')
        sleep(1)

        # terminate monitor worker
        n_workers = len(self.client.nthreads())

        found_worker_dict = self.client.retire_workers(
            names=[self.monitor_process_worker_name],  # name
            close_workers=True,
            remove=True,
        )

        if len(found_worker_dict) == 0:
            found_worker_dict = self.client.retire_workers(
                workers=[self.monitor_process_worker_name],  # address
                close_workers=True,
                remove=True,
            )

        if len(found_worker_dict) > 0:
            while n_workers == len(self.client.nthreads()):
                sleep(1)
            logger.info('Terminate monitor processes worker.')
            sleep(1)
        else:
            logger.warn('Monitor process worker not found.')

        # close FEM (if specified to quit when deconstruct)
        del self.fem
        logger.info('Terminate FEM.')
        sleep(1)

        # close scheduler, other workers(, cluster)
        self.client.shutdown()
        logger.info('Terminate all relative processes.')
        sleep(3)

        # if optimization was crashed, raise Exception
        if self._is_error_exit:
            raise RuntimeError('At least 1 of optimization processes have been crashed. See console log.')


def _start_monitor_server(
        history,
        status,
        worker_addresses,
        worker_status_list,
        host=None,
        port=None,
):
    process_monitor_main(
        history,
        status,
        worker_addresses,
        worker_status_list,
        host,
        port,
    )
    return 'Exit monitor server process gracefully'
