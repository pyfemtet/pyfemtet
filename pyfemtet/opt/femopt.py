from __future__ import annotations

import platform
from typing import Callable, Sequence

import os
import sys
from time import sleep, time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import sympy

import pyfemtet
from pyfemtet._i18n.messages import _
from pyfemtet._util.dask_util import *
from pyfemtet.opt.worker_status import *
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.opt.interface import *
from pyfemtet.opt.optimizer import *
from pyfemtet.opt.optimizer._base_optimizer import DIRECTION, OptimizationDataPerFEM
from pyfemtet.opt.history import History
from pyfemtet.logger import get_module_logger
from pyfemtet.opt.visualization.history_viewer._process_monitor._application import (
    process_monitor_main,
    MonitorHostRecord
)


logger = get_module_logger('opt.femopt', False)


class FEMOpt:
    """
    A class to manage finite element method (FEM) optimization using a specified optimizer and FEM interface.

    Attributes:
        opt (AbstractOptimizer): The optimizer instance to be used for optimization.
        monitor_info (dict[str, str | int | None]): Dictionary to store monitoring information such as host and port.

    Args:
        fem (AbstractFEMInterface, optional): An instance of a FEM interface. Defaults to None, in which case a FemtetInterface is used.
        opt (AbstractOptimizer, optional): An optimizer instance. Defaults to None, in which case OptunaOptimizer is used.

    """

    opt: AbstractOptimizer

    def __init__(
            self,
            fem: AbstractFEMInterface = None,
            opt: AbstractOptimizer = None,
    ):
        self.opt: AbstractOptimizer = opt or OptunaOptimizer()
        
        # fem が与えられていれば opt にセット
        if fem is not None:
            self.opt.fem = fem

        # この時点で opt に fem がセットされていなければ
        # デフォルトをセット
        if len(self.opt.fem_manager.fems) == 0:
            self.opt.fem = FemtetInterface()
        
        # fem が正しくセットされているか確認
        if len(self.opt.fem_manager.fems) == 0:
            raise RuntimeError(
                "FEM interface could not be initialized. "
                "Please ensure that a valid FEM interface is provided or can be created."
            )
        
        self.monitor_info: dict[str, str | int | None] = dict(
            host=None, port=None,
        )

    def add_fem(self, fem: AbstractFEMInterface) -> OptimizationDataPerFEM:
        return self.opt.fem_manager.append(fem)

    def add_constant_value(
            self,
            name: str,
            value: SupportedVariableTypes,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ):
        self.opt.add_constant_value(name, value, properties, pass_to_fem=pass_to_fem)

    def add_parameter(
            self,
            name: str,
            initial_value: float | None = None,
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            step: float | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:
        self.opt.add_parameter(name, initial_value, lower_bound, upper_bound, step, properties, pass_to_fem=pass_to_fem, fix=fix)

    def add_expression_string(
            self,
            name: str,
            expression_string: str,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ) -> None:
        self.opt.add_expression_string(name, expression_string, properties, pass_to_fem=pass_to_fem)

    def add_expression_sympy(
            self,
            name: str,
            sympy_expr: sympy.Expr,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ) -> None:
        self.opt.add_expression_sympy(name, sympy_expr, properties, pass_to_fem=pass_to_fem)

    def add_expression(
            self,
            name: str,
            fun: Callable[..., float],
            properties: dict[str, ...] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            *,
            pass_to_fem: bool = True,
    ) -> None:
        self.opt.add_expression(name, fun, properties, args, kwargs, pass_to_fem=pass_to_fem)

    def add_categorical_parameter(
            self,
            name: str,
            initial_value: SupportedVariableTypes | None = None,
            choices: list[SupportedVariableTypes] | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:
        self.opt.add_categorical_parameter(name, initial_value, choices, properties, pass_to_fem=pass_to_fem, fix=fix)

    def add_objective(
            self,
            name: str,
            fun: Callable[..., float],
            direction: DIRECTION = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> None:
        self.opt.add_objective(name, fun, direction, args, kwargs)

    def add_objectives(
            self,
            names: str | list[str],
            fun: Callable[..., Sequence[float]],
            n_return: int,
            directions: DIRECTION | Sequence[DIRECTION] = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
    ):
        self.opt.add_objectives(names, fun, n_return, directions, args, kwargs)

    def add_constraint(
            self,
            name: str,
            fun: Callable[..., float],
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            strict: bool = True,
            using_fem: bool | None = None,
    ):
        self.opt.add_constraint(name, fun, lower_bound, upper_bound, args, kwargs, strict, using_fem)

    def add_other_output(
            self,
            name: str,
            fun: Callable[..., float],
            args: tuple | None = None,
            kwargs: dict | None = None,
    ):
        self.opt.add_other_output(name, fun, args, kwargs)

    def add_trial(
            self,
            parameters: dict[str, SupportedVariableTypes],
    ):
        self.opt.add_trial(parameters)

    def add_sub_fidelity_model(
            self,
            name: str,
            sub_fidelity_model: SubFidelityModel,
            fidelity: Fidelity,
    ):
        self.opt.add_sub_fidelity_model(name, sub_fidelity_model, fidelity)

    def set_termination_condition(
            self,
            func: Callable[[History], bool] | None,
    ):
        self.opt.set_termination_condition(func)

    def set_monitor_host(self, host: str = None, port: int = None):
        """Sets the host IP address and the port of the process monitor.

        Args:
            host (str):
                The hostname or IP address of the monitor server.
            port (int, optional):
                The port number of the monitor server.
                If None, ``8080`` will be used.
                Defaults to None.

        Tip:
            Specifying host ``0.0.0.0`` allows viewing monitor
            from all computers on the local network.

            If no hostname is specified, the monitor server
            will be hosted on ``localhost``.

            We can access process monitor by accessing
            ```localhost:8080``` on our browser by default.

        """
        if host is not None:
            self.monitor_info.update(host=host)
        if port is not None:
            self.monitor_info.update(port=port)

    def set_random_seed(self, seed: int):
        self.opt.seed = seed

    def optimize(
            self,
            n_trials: int = None,
            n_parallel: int = 1,
            timeout: float = None,
            wait_setup: bool = True,
            confirm_before_exit: bool = True,
            history_path: str = None,
            with_monitor: bool = True,
            scheduler_address: str = None,
            seed: int | None = None,
    ):

        # ===== show initialize info =====
        logger.info(
            _(
                '===== pyfemtet version {ver} =====',
                ver=pyfemtet.__version__,
            )
        )
        client: Client

        # set arguments
        self.opt.n_trials = n_trials or self.opt.n_trials
        self.opt.timeout = timeout or self.opt.timeout
        self.opt.history.path = history_path or self.opt.history.path
        if seed is not None:
            self.opt.seed = seed

        # construct opt workers
        n_using_cluster_workers = n_parallel  # workers excluding main
        if scheduler_address is None:
            n_using_cluster_workers = n_using_cluster_workers - 1
            worker_name_base = 'Sub'
            if n_parallel == 1:
                cluster = nullcontext()
                # noinspection PyTypeChecker
                client = DummyClient()

            else:
                logger.info(_('Launching processes...'))
                cluster = LocalCluster(
                    n_workers=n_parallel - 1,
                    threads_per_worker=1 if n_parallel > 1 else None,
                    processes=True if n_parallel > 1 else False,
                )

                logger.info(_('Connecting cluster...'))
                client = Client(
                    cluster,
                )
        else:
            worker_name_base = 'Remote Worker'
            cluster = nullcontext()
            client = Client(scheduler_address)

        # construct other workers
        main_worker_names = list()
        logger.info(_('Launching threads...'))
        executor_workers = 2  # save_history, watch_status
        if with_monitor:
            executor_workers += 1  # monitor
        if scheduler_address is None:
            executor_workers += 1  # main
            main_worker_names.append('Main')
        executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix='thread_worker'
        )

        with cluster, client, self.opt.history, executor:

            logger.info(_('Setting up...'))

            # finalize history
            self.opt._refresh_problem()
            self.opt.variable_manager.resolve()
            self.opt._finalize_history()

            # optimizer-specific setup after history finalized
            self.opt._setup_before_parallel()

            # setup FEM (mainly for distributing files)
            self.opt.fem_manager.all_fems_as_a_fem._setup_before_parallel()

            # create worker status list
            entire_status = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
            assert n_parallel == len(main_worker_names) + n_using_cluster_workers

            worker_status_list = []
            for i in range(n_parallel):
                worker_status = WorkerStatus(f'worker-status-{i}')
                worker_status.value = WorkerStatus.initializing
                worker_status_list.append(worker_status)
            entire_status.value = WorkerStatus.initializing

            # Get workers and Assign roles
            opt_worker_addresses = list(client.scheduler_info()['workers'].keys())[:n_using_cluster_workers]
            opt_worker_names = [f'{worker_name_base} {i+1}' for i in range(n_using_cluster_workers)]

            # Setting up monitor
            if with_monitor:
                logger.info(_('Launching Monitor...'))

                monitor_host_record = MonitorHostRecord()

                # noinspection PyTypeChecker,PyUnusedLocal
                monitor_future = executor.submit(
                    process_monitor_main,
                    history=self.opt.history,
                    status=entire_status,
                    worker_addresses=main_worker_names + opt_worker_addresses,
                    worker_names=main_worker_names + opt_worker_names,
                    worker_status_list=worker_status_list,
                    host=self.monitor_info['host'],
                    port=self.monitor_info['port'],
                    host_record=monitor_host_record,
                )

            else:
                monitor_future = None
                monitor_host_record = None

            logger.info(_('Setting up optimization problem...'))
            entire_status.value = WorkerStatus.running

            # Run on cluster
            futures = client.map(
                self.opt._run,
                # Arguments of func
                range(n_using_cluster_workers),  # worker_index
                opt_worker_names,  # worker_name
                [self.opt.history] * n_using_cluster_workers,  # history
                [entire_status] * n_using_cluster_workers,  # entire_status
                worker_status_list[:n_using_cluster_workers],  # worker_status
                [worker_status_list] * n_using_cluster_workers,  # worker_status_list
                [wait_setup] * n_using_cluster_workers,  # wait_other_process_setup
                # Arguments of map
                workers=opt_worker_addresses,
                allow_other_workers=False,
            )

            # Run on main process
            assert len(main_worker_names) == 0 or len(main_worker_names) == 1
            if len(main_worker_names) == 1:
                # noinspection PyTypeChecker
                future = executor.submit(
                    self.opt._run,
                    main_worker_names[0],
                    main_worker_names[0],
                    self.opt.history,
                    entire_status,
                    worker_status_list[n_using_cluster_workers:][0],
                    worker_status_list,
                    wait_setup
                )
            else:
                class DummyFuture:
                    def result(self):
                        pass

                future = DummyFuture()

            # Saving history
            def save_history():
                while True:
                    sleep(2)
                    if len(self.opt.history.get_df()) > 0:
                        try:
                            self.opt.history.save()
                            logger.debug('History saved!')
                        except PermissionError:
                            logger.error(
                                _('Cannot save history. '
                                  'The most common reason is '
                                  'that the csv is opened by '
                                  'another program (such as Excel). '
                                  'Please free {path} or lost the '
                                  'optimization history.',
                                  path=self.opt.history.path)
                            )
                    if entire_status.value >= WorkerStatus.finished:
                        break
                logger.debug('History save thread finished!')
            future_saving = executor.submit(save_history, )

            # Watching
            def watch_worker_status():
                while True:
                    sleep(1)
                    logger.debug([s.value for s in worker_status_list])
                    if all([s.value >= WorkerStatus.finished for s in worker_status_list]):
                        break
                if entire_status.value < WorkerStatus.finished:
                    entire_status.value = WorkerStatus.finished
                logger.debug('All workers finished!')
            future_watching = executor.submit(watch_worker_status, )

            if monitor_host_record is not None:
                # update additional_data of history
                # to notify how to emit interruption
                # signal by external processes
                monitor_record_wait_start = time()
                while len(monitor_host_record.get()) == 0:
                    sleep(0.1)
                    if time() - monitor_record_wait_start > 30:
                        logger.warning(_(
                            en_message='Getting monitor host information is '
                                       'failed within 30 seconds. '
                                       'It can not be able to terminate by '
                                       'requesting POST '
                                       '`<host>:<port>/interrupt` '
                                       'by an external process.',
                            jp_message='モニターの情報取得が 30 秒以内に'
                                       '終わりませんでした。最適化プロセスは、'
                                       '外部プロセスから '
                                       '`<host>:<port>/interrupt` に POST を'
                                       'リクエストしても終了できない可能性が'
                                       'あります。'
                        ))
                        break
                if len(monitor_host_record.get()) > 0:
                    self.opt.history.additional_data.update(
                        monitor_host_record.get()
                    )

            # Terminating monitor even if exception is raised
            class TerminatingMonitor:

                def __init__(self, monitor_future_):
                    self.monitor_future_ = monitor_future_

                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_val, exc_tb):
                    # Send termination signal to monitor
                    # and wait to finish
                    # noinspection PyTypeChecker
                    entire_status.value = WorkerStatus.terminated
                    if self.monitor_future_ is not None:
                        self.monitor_future_.result()

            with TerminatingMonitor(monitor_future):
                # Wait to finish optimization
                client.gather(futures)
                future.result()
                future_saving.result()
                future_watching.result()

            if confirm_before_exit:
                confirm_msg = _(
                    en_message='The optimization is now complete. '
                               'You can view the results on the monitor '
                               'until you press Enter to exit the program.',
                    jp_message='最適化が終了しました。'
                               'プログラムを終了するまで、'
                               '結果をプロセスモニターで確認できます。'
                )
                result_viewer_msg = _(
                    en_message='After the program ends, '
                               'you can check the optimization results '
                               'using the result viewer.\n'
                               'The result viewer can be launched by '
                               'performing one of the following actions:\n'
                               '- {windows_only}Launch the `pyfemtet-opt-result-viewer` '
                               'shortcut on your desktop if exists.\n'
                               '- {windows_only}Launch {dir}.\n'
                               '- Execute "py -m pyfemtet.opt.visualization.history_viewer" '
                               'in the command line',
                    jp_message='プログラム終了後も、結果ビューワを使って最適化結果を'
                               '確認することができます。'
                               '結果ビューワは以下のいずれかを実施すると起動できます。\n'
                               '- {windows_only}デスクトップの pyfemtet-opt-result-viewer '
                               'ショートカットを起動する\n'
                               '- {windows_only}{dir} にある {filename} を起動する\n'
                               '- コマンドラインで「py -m pyfemtet.opt.visualization.history_viewer」'
                               'を実行する',
                    dir=os.path.abspath(os.path.dirname(sys.executable)),
                    filename='pyfemtet-opt-result-viewer.exe (or .cmd)',
                    windows_only='(Windows only) ' if platform.system() != 'Windows' else '',
                )
                print("====================")
                print(confirm_msg)
                print(result_viewer_msg)
                print(_(
                    en_message='Press Enter to quit...',
                    jp_message='終了するには Enter を押してください...',
                ))
                input()

            df = self.opt.history.get_df()

        logger.info(_('All processes are terminated.'))

        return df


def debug_1():
    # noinspection PyUnresolvedReferences
    from time import sleep
    # from pyfemtet.opt.optimizer import InterruptOptimization
    import optuna
    from pyfemtet.opt.interface import AbstractFEMInterface, NoFEM

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        # print(os.getpid())
        # raise RuntimeError
        # raise Interrupt
        # if get_worker() is None:
        #     raise RuntimeError
        return (x ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return x[0]

    _opt = OptunaOptimizer()
    _opt.sampler = optuna.samplers.TPESampler(seed=42)
    _opt.n_trials = 10

    _fem = NoFEM()
    _opt.fem = _fem

    _args = (_opt,)

    _opt.add_parameter('x1', 1, -1, 1, step=0.2)
    _opt.add_parameter('x2', 1, -1, 1, step=0.2)

    _opt.add_constraint('cns', _cns, lower_bound=-0.5, args=_args)

    _opt.add_objective('obj', _parabola, args=_args)

    _femopt = FEMOpt(fem=_fem, opt=_opt)
    _femopt.opt = _opt
    # _femopt.opt.history.path = 'v1test/femopt-restart-test.csv'
    _femopt.optimize(n_parallel=2)

    print(os.path.abspath(_femopt.opt.history.path))


def substrate_size(Femtet):
    """基板のXY平面上での専有面積を計算します。"""
    substrate_w = Femtet.GetVariableValue('substrate_w')
    substrate_d = Femtet.GetVariableValue('substrate_d')

    # assert get_worker() is not None

    return substrate_w * substrate_d  # 単位: mm2


def debug_2():
    from pyfemtet.opt.interface import FemtetInterface
    from pyfemtet.opt.optimizer import OptunaOptimizer

    fem = FemtetInterface(
        femprj_path=os.path.join(os.path.dirname(__file__), 'wat_ex14_parametric_jp.femprj'),
    )

    opt = OptunaOptimizer()

    opt.fem = fem

    opt.add_parameter(name="substrate_w", initial_value=40, lower_bound=22, upper_bound=60)
    opt.add_parameter(name="substrate_d", initial_value=60, lower_bound=34, upper_bound=60)
    opt.add_objective(name='基板サイズ(mm2)', fun=substrate_size)
    opt.add_objective(name='obj2', fun=substrate_size)
    opt.add_objective(name='obj3', fun=substrate_size)

    opt.n_trials = 10
    # opt.history.path = os.path.join(os.path.dirname(__file__), 'femtet-test.csv')

    femopt = FEMOpt()

    femopt.opt = opt

    femopt.optimize(n_parallel=1)


def debug_3():
    # noinspection PyUnresolvedReferences
    from time import sleep
    # from pyfemtet.opt.optimizer import InterruptOptimization
    import optuna
    from pyfemtet.opt.interface import AbstractFEMInterface, NoFEM

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        # print(os.getpid())
        # raise RuntimeError
        # raise Interrupt
        # if get_worker() is None:
        #     raise RuntimeError
        return (x ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return x[0]

    _opt = OptunaOptimizer()
    _opt.sampler = optuna.samplers.TPESampler(seed=42)

    _fem = NoFEM()
    _opt.fem = _fem

    _args = (_opt,)

    _opt.add_parameter('x1', 1, -1, 1, step=0.2)
    _opt.add_parameter('x2', 1, -1, 1, step=0.2)

    _opt.add_constraint('cns', _cns, lower_bound=-0.5, args=_args)

    _opt.add_objective('obj', _parabola, args=_args)

    _femopt = FEMOpt(fem=_fem, opt=_opt)
    _femopt.optimize(
        scheduler_address='<dask scheduler で起動したスケジューラの tcp をここに入力>',
        n_trials=80,
        n_parallel=6,
        with_monitor=True,
        confirm_before_exit=False,
    )


if __name__ == '__main__':
    # for i in range(1):
        debug_1()
    # debug_2()
    # debug_3()
