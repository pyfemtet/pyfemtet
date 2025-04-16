import os
from time import sleep
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import _pyfemtet

from pyfemtet._i18n import _
from pyfemtet._util.dask_util import *
from pyfemtet.opt.optimizer import *
from pyfemtet.opt.worker_status import *
from pyfemtet.logger import get_module_logger
from pyfemtet.opt.visualization.history_viewer._process_monitor._application import process_monitor_main


logger = get_module_logger('opt.femopt', False)


class FEMOpt:
    opt: AbstractOptimizer

    def optimize(
            self,
            n_parallel: int = 1,
            with_monitor: bool = True,
    ) -> None:

        logger.info(
            _(
                '===== pyfemtet version {ver} =====',
                ver=_pyfemtet.__version__,
            )
        )
        client: Client
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

        logger.info(_('Launching threads...'))
        executor_workers = 3  # main, save_history, watch_status
        if with_monitor:
            executor_workers += 1
        executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix='thread_worker'
        )

        with cluster, client, self.opt.history, executor:

            logger.info(_('Setting up...'))

            # finalize history
            self.opt._load_problem_from_fem()
            self.opt._finalize_history()

            # optimizer-specific setup after history finalized
            self.opt._setup_before_parallel()

            # setup FEM (mainly for distributing files)
            self.opt.fem._setup_before_parallel()

            # create worker status list
            entire_status = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
            worker_status_list = [WorkerStatus(f'worker-status-{i}') for i in range(n_parallel)]
            entire_status.value = WorkerStatus.initializing

            # Get workers
            nannies: tuple[Nanny] = tuple(client.cluster.workers.values())

            # Assign roles
            opt_worker_addresses = [n.worker_address for n in nannies]

            # Setting up monitor
            if with_monitor:
                logger.info(_('Launching Monitor...'))
                # noinspection PyTypeChecker,PyUnusedLocal
                monitor_future = executor.submit(
                    process_monitor_main,
                    history=self.opt.history,
                    status=entire_status,
                    worker_addresses=['main'] + opt_worker_addresses,
                    worker_status_list=worker_status_list,
                )
            else:
                monitor_future = None

            logger.info(_('Setting up optimization problem...'))
            entire_status.value = WorkerStatus.running

            # Run on cluster
            futures = client.map(
                self.opt._run,
                # Arguments of func
                range(1, n_parallel),
                [self.opt.history] * (n_parallel - 1),
                [entire_status] * (n_parallel - 1),
                worker_status_list[1:],
                [worker_status_list] * (n_parallel - 1),
                # Arguments of map
                workers=opt_worker_addresses,
                allow_other_workers=False,
            )

            # Run on main process
            # noinspection PyTypeChecker
            future = executor.submit(
                self.opt._run,
                'Main',
                self.opt.history,
                entire_status,
                worker_status_list[0],
                worker_status_list,
            )

            # Saving history
            def save_history():
                while True:
                    sleep(2)
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

        logger.info(_('All processes are terminated.'))


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

    _args = (_fem, _opt)

    _opt.add_parameter('x1', 1, -1, 1, step=0.2)
    _opt.add_parameter('x2', 1, -1, 1, step=0.2)

    _opt.add_constraint('cns', _cns, lower_bound=-0.5, args=_args)

    _opt.add_objective('obj', _parabola, args=_args)

    _femopt = FEMOpt()
    _femopt.opt = _opt
    _femopt.opt.history.path = 'v1test/femopt-restart-test.csv'
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


if __name__ == '__main__':
    # for i in range(1):
    #     debug_1()
    debug_2()
