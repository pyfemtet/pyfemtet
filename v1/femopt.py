import os
from concurrent.futures import ThreadPoolExecutor

import pyfemtet

from v1.dask_util import *
from v1.optimizer import *
from v1.monitor import *
from v1.worker_status import *
from v1.logger import get_module_logger

logger = get_module_logger('opt.femopt', False)


class FEMOpt:
    opt: AbstractOptimizer

    def optimize(self, n_parallel, path=None) -> None:

        self.opt.history.path = path

        logger.info(f'===== pyfemtet version {pyfemtet.__version__} =====')

        logger.info(f'Launching processes...')
        _cluster = LocalCluster(
            n_workers=n_parallel - 1 + 1,
            threads_per_worker=1 if n_parallel > 1 else None,
            processes=True if n_parallel > 1 else False,
        )

        logger.info(f'Connecting cluster...')
        _client = Client(
            _cluster,
        )

        logger.info(f'Launching threads...')
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='thread_worker'
        )

        with _cluster, _client as client, self.opt.history:

            logger.info(f'Setting up...')

            # finalize history
            self.opt._finalize_history()

            # create worker status list
            entire_status = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
            worker_status_list = [WorkerStatus() for _ in range(n_parallel)]
            entire_status.value = WorkerStatus.initializing

            # Get workers
            nannies: tuple[Nanny] = tuple(client.cluster.workers.values())

            # Assign roles
            monitor_worker_address = nannies[0].worker_address
            opt_worker_addresses = [n.worker_address for n in nannies[1:]]

            # Setting up monitor
            logger.info(f'Launching Monitor...')
            monitor_future = client.submit(
                run_monitor,
                # Arguments of func
                history=self.opt.history,
                entire_status=entire_status,
                worker_status_list=worker_status_list,
                # Arguments of submit
                workers=(monitor_worker_address,),
                allow_other_workers=False,
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
                range(n_parallel - 1),
                [self.opt.history] * (n_parallel - 1),
                [entire_status] * (n_parallel - 1),
                worker_status_list[1:],
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
            entire_status.value = WorkerStatus.terminated
            monitor_future.result()

            self.opt.history.save()

        logger.info('All processes are terminated.')


def test():
    # from time import sleep
    # from v1.optimizer import InterruptOptimization
    from v1.interface import AbstractFEMInterface, NoFEM

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        # sleep(1)
        print(os.getpid())
        # raise RuntimeError
        # raise Interrupt
        return (x ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return x[0]

    _opt = OptunaOptimizer()

    _fem = NoFEM()
    _opt.fem = _fem

    _args = (_fem, _opt)

    _opt.add_parameter('x1', 1, -1, 1, step=0.5)
    _opt.add_parameter('x2', 1, -1, 1, step=0.5)

    _opt.add_constraint('cns', _cns, lower_bound=0, args=_args)

    _opt.add_objective('obj', _parabola, args=_args)

    _femopt = FEMOpt()
    _femopt.opt = _opt
    _femopt.optimize(n_parallel=1, path='femopt-restart-test.csv')

    print(os.path.abspath(_femopt.opt.history.path))


if __name__ == '__main__':
    test()
