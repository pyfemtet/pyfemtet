import os
from time import sleep
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import _pyfemtet

from v1.utils.dask_util import *
from v1.optimizer import *
from v1.worker_status import *
from v1.logger import get_module_logger
from v1.visualization2.monitor_application._process_monitor.application import main


logger = get_module_logger('opt.femopt', True)


class FEMOpt:
    opt: AbstractOptimizer

    def optimize(self, n_parallel) -> None:

        logger.info(f'===== pyfemtet version {_pyfemtet.__version__} =====')
        client: Client
        if n_parallel == 1:
            cluster = nullcontext()
            # noinspection PyTypeChecker
            client = DummyClient()
        else:
            logger.info(f'Launching processes...')
            cluster = LocalCluster(
                n_workers=n_parallel - 1,
                threads_per_worker=1 if n_parallel > 1 else None,
                processes=True if n_parallel > 1 else False,
            )

            logger.info(f'Connecting cluster...')
            client = Client(
                cluster,
            )

        logger.info(f'Launching threads...')
        executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix='thread_worker'
        )

        with cluster, client, self.opt.history, executor:

            logger.info(f'Setting up...')

            # finalize history
            self.opt._finalize_history()

            # finalize optimizer after history finalized
            self.opt._setup_before_parallel()

            # create worker status list
            entire_status = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
            worker_status_list = [WorkerStatus() for _ in range(n_parallel)]
            entire_status.value = WorkerStatus.initializing

            # Get workers
            nannies: tuple[Nanny] = tuple(client.cluster.workers.values())

            # Assign roles
            opt_worker_addresses = [n.worker_address for n in nannies]

            # Setting up monitor
            logger.info(f'Launching Monitor...')
            # noinspection PyTypeChecker,PyUnusedLocal
            monitor_future = executor.submit(
                main,
                history=self.opt.history,
                status=entire_status,
                worker_addresses=['main'] + opt_worker_addresses,
                worker_status_list=worker_status_list,
            )

            logger.info(f'Setting up optimization problem...')
            entire_status.value = WorkerStatus.running

            # Run on main process
            future = executor.submit(
                self.opt._run,
                'Main',
                self.opt.history,
                entire_status,
                worker_status_list[0]
            )

            # Run on cluster
            futures = client.map(
                self.opt._run,
                # Arguments of func
                range(1, n_parallel),
                [self.opt.history] * (n_parallel - 1),
                [entire_status] * (n_parallel - 1),
                worker_status_list[1:],
                # Arguments of map
                workers=opt_worker_addresses,
                allow_other_workers=False,
            )

            # Saving history
            def save_history():
                while True:
                    sleep(2)
                    try:
                        self.opt.history.save()
                        logger.debug('History saved!')
                    except PermissionError:
                        logger.error("書き込みできません。")
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

                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_val, exc_tb):
                    # Send termination signal to monitor
                    # and wait to finish
                    # noinspection PyTypeChecker
                    entire_status.value = WorkerStatus.terminated
                    monitor_future.result()

            with TerminatingMonitor():
                # Wait to finish optimization
                client.gather(futures)
                future.result()
                future_saving.result()
                future_watching.result()

        logger.info('All processes are terminated.')


def test():
    # noinspection PyUnresolvedReferences
    from time import sleep
    # from v1.optimizer import InterruptOptimization
    import optuna
    from v1.interface import AbstractFEMInterface, NoFEM

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


if __name__ == '__main__':
    for i in range(1):
        test()
