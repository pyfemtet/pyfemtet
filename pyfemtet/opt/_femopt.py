# built-in
import inspect
import warnings
from typing import Optional, Any, Callable, List, Sequence, SupportsFloat
import os
import datetime
from time import time, sleep
from threading import Thread
import json
from traceback import print_exception

# 3rd-party
import numpy as np
import pandas as pd
from dask.distributed import LocalCluster, Client

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface, FemtetInterface
from pyfemtet.opt.optimizer import AbstractOptimizer, OptunaOptimizer
from pyfemtet.opt.visualization._process_monitor.application import main as process_monitor_main
from pyfemtet.opt._femopt_core import (
    _check_bound,
    _is_access_gogh,
    _is_access_femtet,
    Objective,
    Constraint,
    History,
    OptimizationStatus,
    logger,
)
from pyfemtet._message import Msg, encoding
from pyfemtet.opt.optimizer.parameter import Parameter, Expression
from pyfemtet._warning import experimental_feature

from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None})


def add_worker(client, worker_name, n_workers=1):
    import sys
    from subprocess import Popen, DEVNULL, PIPE

    current_n_workers = len(client.nthreads().keys())

    Popen(
        f'{sys.executable} -m dask worker '
        f'{client.scheduler.address} '
        f'--nthreads 1 '
        f'--nworkers {n_workers} '
        f'--name {worker_name} ',  # A unique name for this worker like ‘worker-1’. If used with -nworkers then the process number will be appended like name-0, name-1, name-2, …
        # --no-nanny option は --nworkers と併用できない
        shell=True,
        stderr=DEVNULL,
        stdout=DEVNULL,
    )

    # worker が増えるまで待つ
    client.wait_for_workers(n_workers=current_n_workers + n_workers)


class FEMOpt:
    """Class to control FEM interface and optimizer.

    Args:
        fem (FEMInterface, optional):
            The FEM software interface.
            Defaults to None (automatically set to :class:`FemtetInterface`).

        opt (AbstractOptimizer, optional):
            The numerical optimizer object.
            Defaults to None (automatically set to :class:`OptunaOptimizer`
            with :class:`optuna.samplers.TPESampler`).

        history_path (str, optional):
            The path to the history file.
            Defaults to None.
            If None, '%Y_%m_%d_%H_%M_%S.csv' is created in current directory.

        scheduler_address (str, optional):
            When performing cluster computing, please specify the address
            of the scheduler computer here.

            See Also:
                https://pyfemtet.readthedocs.io/en/stable/pages/usage_pages/how_to_deploy_cluster.html

    .. Example の中について、reST ではリストだと改行できないので、リストにしない

    Examples:

        When specifying and opening a femprj and model, we write

            >>> from pyfemtet.opt import FEMOpt, FemtetInterface  # doctest: +SKIP
            >>> fem = FemtetInterface(femprj_path='path/to/project.femprj', model_name='NewModel')  # doctest: +SKIP
            >>> femopt = FEMOpt(fem=fem)  # doctest: +SKIP

        When specifying optimization algorithm, we write

            >>> from optuna.samplers import TPESampler  # doctest: +SKIP
            >>> from pyfemtet.opt import FEMOpt, OptunaOptimizer  # doctest: +SKIP
            >>> opt = OptunaOptimizer(sampler_class=TPESampler, sampler_kwargs=dict(n_startup_trials=10))  # doctest: +SKIP
            >>> femopt = FEMOpt(opt=opt)  # doctest: +SKIP

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
            self.opt: AbstractOptimizer = OptunaOptimizer()
        else:
            self.opt: AbstractOptimizer = opt

        # メンバーの宣言
        self.client = None
        self.status = None  # actor
        self.history = None  # actor
        self.worker_status_list = None  # [actor]
        self.monitor_process_future = None
        self.monitor_server_kwargs = dict()
        self.monitor_process_worker_name = None
        self._hv_reference = None

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
            initial_value: float = None,
            lower_bound: float = None,
            upper_bound: float = None,
            step: float = None,
            properties: dict[str, str or float] = None,
            pass_to_fem: bool = True,
    ):
        # noinspection PyUnresolvedReferences
        """Adds a parameter to the optimization problem.

        Args:
            name (str): The name of the parameter.
            initial_value (float, optional): The initial value of the parameter. Defaults to None. If None, try to get initial value from FEMInterface.
            lower_bound (float, optional): The lower bound of the parameter. Defaults to None. Some optimization algorithms require this.
            upper_bound (float or None, optional): The upper bound of the parameter. Defaults to None. Some optimization algorithms require this.
            step (float, optional): The step of parameter. If specified, parameter is used as discrete. Defaults to None.
            properties (dict[str, str or float], optional): Additional information about the parameter. Defaults to None.
            pass_to_fem (bool, optional): If this variable is used directly in FEM model update or not. If False, this parameter can be just used as inpt of expressions. Defaults to True.

        Raises:
            ValueError: If initial_value is not specified and the value for the given name is also not specified in FEM.

        Examples:

            When adding parameter a (-1 <= a <= 1; initial value is 0), we write

                >>> femopt.add_parameter('parameter_a', 0, -1, 1)  # doctest: +SKIP

                Note that the ```note``` argument can be set any name in this case.

            When adding discrete parameter a (-1 <= a <= 1; initial value is 0,
            step 0.5), we write

                >>> femopt.add_parameter('parameter a', 0, -1, 1, 0.5)  # doctest: +SKIP

        """

        _check_bound(lower_bound, upper_bound, name)

        if pass_to_fem:
            value = self.fem.check_param_value(name)
            if initial_value is None:
                if value is not None:
                    initial_value = value
                else:
                    raise ValueError('initial_value を指定してください.')
        else:
            if initial_value is None:
                raise ValueError('initial_value を指定してください.')

        prm = Parameter(
            name=name,
            value=float(initial_value),
            lower_bound=float(lower_bound) if lower_bound is not None else None,
            upper_bound=float(upper_bound) if upper_bound is not None else None,
            step=float(step) if step is not None else None,
            pass_to_fem=pass_to_fem,
            properties=properties,
        )
        self.opt.variables.add_parameter(prm)

    @experimental_feature
    def add_expression(
            self,
            name: str,
            fun: Callable[[Any], float],
            properties: dict[str, str or float] = None,
            kwargs: Optional[dict] = None,
            pass_to_fem=True,
    ):
        # noinspection PyUnresolvedReferences
        """Add expression to the optimization problem.

        Warnings:
            This feature is highly experimental and
            may change in the future.

        Args:
            name (str): The name of the variable.
            fun (Callable[[Any], float]): An expression function. The arguments that you want to use as input variables must be the same with ``name`` of Variable objects added by ``add_parameter()`` or ``add_expression()``. If you use other objects as argument of the function, you must specify ``kwargs``.
            properties (dict[str, str or float], optional): Additional information about the parameter. Defaults to None.
            kwargs (Optional[dict], optional): Remaining arguments of ``fun``. Defaults to None.
            pass_to_fem (bool, optional): If this variable is used directly in FEM model update or not. If False, this variable can be just used as inpt of other expressions. Defaults to True.

        Examples:

            When adding variable a and b; a is not directly included as model
            variables but an optimization parameter, b is not a parameter but
            a model variable and the relationship of these 2 variables is
            ```b = a ** 2```, we write:

                >>> def calc_b(parameter_a):  # doctest: +SKIP
                ...     return parameter_a ** 2  # doctest: +SKIP
                ...
                >>> femopt.add_parameter('parameter_a', 0, -1, 1, pass_to_fem=False)  # doctest: +SKIP
                >>> femopt.add_expression('variable_b', calc_b)  # doctest: +SKIP

                Notes:
                    The argument names of function to calculate variable
                    must match the ```name``` of the parameter.

                Notes:
                    In this case, only strings that can be used as Python variables are valid
                    for ```name``` argument of :func:`FEMOpt.add_parameter`.

            When adding variable r, theta and x, y; r, theta is an optimization
            parameter and x, y is an intermediate variable to calculate constraint
            function, we write

                >>> def calc_x(r, theta):  # doctest: +SKIP
                ...     return r * cos(theta)  # doctest: +SKIP
                ...
                >>> def calc_y(r, theta):  # doctest: +SKIP
                ...     return r * cos(theta)  # doctest: +SKIP
                ...
                >>> def constraint(Femtet, opt: AbstractOptimizer):  # doctest: +SKIP
                ...     d = opt.variables.get_variables()  # doctest: +SKIP
                ...     x, y = d['x'], d['y']  # doctest: +SKIP
                ...     return min(x, y)  # doctest: +SKIP
                ...
                >>> femopt.add_parameter('r', 0, 0, 1)  # doctest: +SKIP
                >>> femopt.add_parameter('theta', 0, 0, 2*pi)  # doctest: +SKIP
                >>> femopt.add_expression('x', calc_x, pass_to_fem=False)  # doctest: +SKIP
                >>> femopt.add_expression('y', calc_y, pass_to_fem=False)  # doctest: +SKIP
                >>> femopt.add_constraint(constraint, lower_bound=0.5, args=(femopt.opt,))  # doctest: +SKIP

        """

        exp = Expression(
            name=name,
            value=None,
            properties=properties,
            fun=fun,
            kwargs=kwargs if kwargs else {},
            pass_to_fem=pass_to_fem,
        )
        self.opt.variables.add_expression(exp)

    def add_objective(
            self,
            fun,
            name: str or None = None,
            direction: str or float = 'minimize',
            args: tuple or None = None,
            kwargs: dict or None = None
    ):
        # noinspection PyUnresolvedReferences
        """Adds an objective to the optimization problem.

        Args:
            fun (callable): The objective function.
            name (str or None, optional): The name of the objective. Defaults to None.
            direction (str or float, optional): The optimization direction. Varid values are 'maximize', 'minimize' or a float value. Defaults to 'minimize'.
            args (tuple or None, optional): Additional arguments for the objective function. Defaults to None.
            kwargs (dict or None, optional): Additional keyword arguments for the objective function. Defaults to None.

        Note:
            If the FEMInterface is FemtetInterface, the 1st argument of fun should be Femtet (IPyDispatch) object.

        Examples:

            When add a complex objective (for example, want to dynamically set body
            to calculate temperature and use product of the temperature and one of
            the parameters as objective) , we write

                >>> class MyClass:  # doctest: +SKIP
                ...     def get_body_name(self):  # doctest: +SKIP
                ...         ...  # process something and detect body name  # doctest: +SKIP
                ...         return body_name  # doctest: +SKIP
                ...
                >>> def complex_objective(Femtet, opt, some_object):  # doctest: +SKIP
                ...     body_name = some_object.get_body_name()  # dynamically get body name to calculate temperature  # doctest: +SKIP
                ...     temp, _, _ = Femtet.Gogh.Watt.GetTemp(body_name)  # calculate temperature  # doctest: +SKIP
                ...     some_param = opt.get_parameter()['some_param']  # get ome parameter  # doctest: +SKIP
                ...     return temp * some_param  # calculate something and return it  # doctest: +SKIP
                ...
                >>> my_obj = MyClass()  # doctest: +SKIP
                >>> femopt.add_objective(complex_objective, args=(femopt.opt, my_obj,))  # doctest: +SKIP

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

    def add_objectives(
            self,
            fun: Callable[[Any], Sequence[SupportsFloat]],
            n_return: int,
            names: str or Sequence[str] or None = None,
            directions: str or Sequence[str] or None = None,
            args: tuple or None = None,
            kwargs: dict or None = None,
    ):
        from pyfemtet.opt._femopt_core import ObjectivesFunc
        components = ObjectivesFunc(fun, n_return)

        if names is not None:
            if isinstance(names, str):
                names = [f'name_{i}' for i in range(n_return)]
            else:
                # names = names
                pass
        else:
            names = [None for _ in range(n_return)]

        if directions is not None:
            if isinstance(directions, str):
                directions = [directions for _ in range(n_return)]
            else:
                # directions = directions
                pass
        else:
            directions = ['minimize' for _ in range(n_return)]

        for name, component, direction in zip(names, components, directions):
            self.add_objective(
                fun=component,
                name=name,
                direction=direction,
                args=args,
                kwargs=kwargs,
            )


    def add_constraint(
            self,
            fun,
            name: str or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            strict: bool = True,
            args: tuple or None = None,
            kwargs: dict or None = None,
            using_fem: bool = None,
    ):
        # noinspection PyUnresolvedReferences
        """Adds a constraint to the optimization problem.

        Args:
            fun (callable): The constraint function.
            name (str or None, optional): The name of the constraint. Defaults to None.
            lower_bound (float or Non, optional): The lower bound of the constraint. Defaults to None.
            upper_bound (float or Non, optional): The upper bound of the constraint. Defaults to None.
            strict (bool, optional): Flag indicating if it is a strict constraint. Defaults to True.
            args (tuple or None, optional): Additional arguments for the constraint function. Defaults to None.
            kwargs (dict): Additional arguments for the constraint function. Defaults to None.
            using_fem (bool, optional): Using FEM or not in the constraint function. It may make the processing time in strict constraints in BoTorchSampler. Defaults to None. If None, PyFemtet checks the access to Femtet and estimate using Femtet or not automatically.

        Warnings:

            When ```strict``` == True and using OptunaOptimizer along with :class:`PoFBoTorchSampler`,
            PyFemtet will solve an optimization subproblem to propose new variables.
            During this process, the constraint function ```fun``` will be executed at each
            iteration of the subproblem, which may include time-consuming operations such
            as retrieving 3D model information via FEMInterface.
            As a result, it may not be feasible to complete the overall optimization within
            a realistic timeframe.

            Examples:
                For example, in a case where the bottom area of the model is constrained,
                the following approach may require a very long time.

                    >>> def bottom_area(Femtet, opt):  # doctest: +SKIP
                    ...     w = Femtet.GetVariableValue('width')  # doctest: +SKIP
                    ...     d = Femtet.GetVariableValue('depth')  # doctest: +SKIP
                    ...     return w * d  # doctest: +SKIP
                    ...
                    >>> femopt.add_constraint(constraint, lower_bound=5)  # doctest: +SKIP

                Instead, please do the following.

                    >>> def bottom_area(_, opt):  # Not to access the 1st argument.  # doctest: +SKIP
                    ...     params = opt.get_parameter()  # doctest: +SKIP
                    ...     w, d = params['width'], params['depth']  # doctest: +SKIP
                    ...     return w * d  # doctest: +SKIP
                    ...
                    >>> femopt.add_constraint(constraint, lower_bound=5, args=(femopt.opt,))  # doctest: +SKIP

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
        if using_fem is None:
            # 自動推定機能は Femtet 特有の処理とする
            if isinstance(self.fem, FemtetInterface):
                using_fem = _is_access_femtet(fun)
            else:
                using_fem = False

        # strict constraint の場合、solve 前に評価したいので Gogh へのアクセスを禁ずる
        if strict and isinstance(self.fem, FemtetInterface):
            if _is_access_gogh(fun):
                message = Msg.ERR_CONTAIN_GOGH_ACCESS_IN_STRICT_CONSTRAINT
                raise Exception(message)

        # strict constraint かつ BoTorchSampler の場合、
        # 最適化実行時に monkey patch を実行するフラグを立てる。
        # ただし using_fem が True ならば非常に低速なので警告を出す。
        if strict and isinstance(self.opt, OptunaOptimizer):
            self.opt: OptunaOptimizer
            from optuna_integration import BoTorchSampler
            if issubclass(self.opt.sampler_class, BoTorchSampler):
                # パッチ実行フラグ
                self.opt._do_monkey_patch = True
                # 警告
                if using_fem:
                    logger.warning(Msg.WARN_UPDATE_FEM_PARAMETER_TOOK_A_LONG_TIME)

        self.opt.constraints[name] = Constraint(fun, name, lower_bound, upper_bound, strict, args, kwargs, using_fem)

    def get_parameter(self, format='dict'):
        """Deprecated method.

        Planed to remove in future version. Use FEMOpt.opt.get_parameter() instead.
        """
        raise DeprecationWarning('FEMOpt.get_parameter() was deprecated. Use FEMOpt.opt.get_parameter() instead.')

    def set_monitor_host(self, host=None, port=None):
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
        self.monitor_server_kwargs = dict(
            host=host,
            port=port
        )

    def optimize(
            self,
            n_trials: int = None,
            n_parallel: int = 1,
            timeout: float = None,
            wait_setup: bool = True,
            confirm_before_exit: bool = True,
    ):
        """Runs the main optimization process.

        Args:
            n_trials (int, optional):
                The number of trials.
                Defaults to None.
            n_parallel (int, optional):
                The number of parallel processes.
                Defaults to 1.
                Note that even if this argument is 1,
                :class:`FEMOpt` makes some child processes
                to run process monitor, status monitor, etc.
            timeout (float, optional):
                The maximum amount of time in seconds
                that each trial can run.
                Defaults to None.
            wait_setup (bool, optional):
                Wait for all workers launching FEM system.
                Defaults to True.
            confirm_before_exit (bool, optional):
                Insert stop before exit to continue to
                show process monitor.

        Tip:
            If set_monitor_host() is not executed,
            a local server for monitoring will be
            started at localhost:8080.

            See Also:
                :func:`FEMOpt.set_monitor_host`

        Note:
            If ``n_trials`` and ``timeout`` are both None,
            it runs forever until interrupting by the user.

        Note:
            If ``n_parallel`` >= 2, depending on the end timing,
            ``n_trials`` may be exceeded by up to ``n_parallel-1`` times.

        Warning:
            If ``n_parallel`` >= 2 and ``fem`` is a subclass of
            ``FemtetInterface``, the ``strictly_pid_specify`` of
            subprocess is set to ``False``.
            So **it is recommended to close all other Femtet processes
            before running.**

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
        prm: Parameter = None
        for prm in self.opt.variables.parameters.values():
            lb, ub = prm.lower_bound, prm.upper_bound
            is_incomplete_bounds = is_incomplete_bounds + (lb is None) + (ub is None)
        if is_incomplete_bounds:
            self.opt.method_checker.check_incomplete_bounds()

        # 共通引数
        self.opt.n_trials = n_trials
        self.opt.timeout = timeout

        # resolve expression dependencies
        self.opt.variables.resolve()
        self.opt.variables.evaluate()

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
            add_worker(self.client, self.monitor_process_worker_name)

        else:
            # ローカルクラスターを構築
            logger.info('Launching single machine cluster... This may take tens of seconds.')

            # Fixed:
            #   Nanny の管理機能は必要ないが、Python API では worker_class を Worker にすると
            #   processes 引数が無視されて Thread worker が立てられる。
            #   これは CLI の --no-nanny オプションも同様らしい。

            # クラスターの構築
            cluster = LocalCluster(
                processes=True,
                n_workers=n_parallel,
                threads_per_worker=1,
            )
            logger.info('LocalCluster launched successfully.')

            self.client = Client(cluster, direct_to_workers=False)
            self.scheduler_address = self.client.scheduler.address
            logger.info('Client launched successfully.')

            # 最適化タスクを振り分ける worker を指定
            subprocess_indices = list(range(n_parallel))[1:]
            worker_addresses = list(self.client.nthreads().keys())

            # monitor worker の設定
            self.monitor_process_worker_name = worker_addresses[0]
            worker_addresses[0] = 'Main'

        with self.client.cluster as _cluster, self.client as _client:

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

                # Femtet の confirm_before_exit のセット
                self.fem.confirm_before_exit = confirm_before_exit
                self.fem.kwargs['confirm_before_exit'] = confirm_before_exit

                logger.info('Femtet loaded successfully.')


            # actor の設定
            self.status = OptimizationStatus(_client)
            self.worker_status_list = [OptimizationStatus(_client, name) for name in worker_addresses]  # tqdm 検討
            self.status.set(OptimizationStatus.SETTING_UP)
            self.history = History(
                self.history_path,
                self.opt.variables.get_parameter_names(),
                list(self.opt.objectives.keys()),
                list(self.opt.constraints.keys()),
                _client,
                metadata,
                self._hv_reference
            )
            logger.info('Status Actor initialized successfully.')

            # launch monitor
            # noinspection PyTypeChecker
            self.monitor_process_future = _client.submit(
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
            logger.info('Process monitor initialized successfully.')

            # fem
            self.fem._setup_before_parallel(_client)

            # opt
            self.opt.fem_class = type(self.fem)
            self.opt.fem_kwargs = self.fem.kwargs
            self.opt.entire_status = self.status
            self.opt.history = self.history
            self.opt._setup_before_parallel()

            # クラスターでの計算開始
            self.status.set(OptimizationStatus.LAUNCHING_FEM)
            start = time()
            calc_futures = _client.map(
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

            # ===== 終了 =====

            # クラスターの Unexpected Exception のリストを取得
            opt_exceptions: list[Exception or None] = _client.gather(calc_futures)  # gather() で終了待ちも兼ねる

            # ローカルの opt で計算している場合、その Exception も取得
            local_opt_exception: Exception or None = None
            if not self.opt.is_cluster:
                if t_main is not None:
                    t_main.join()  # 終了待ち
                    local_opt_exception = self.opt._exception  # Exception を取得
            opt_exceptions.append(local_opt_exception)

            # 終了
            self.status.set(OptimizationStatus.TERMINATED)
            end = time()

            # 一応
            t_save_history.join()

            # 結果通知
            logger.info(Msg.OPTIMIZATION_FINISHED)
            logger.info(self.history.path)

            # monitor worker を終了する準備
            # 実際の終了は monitor worker の終了時
            self.status.set(OptimizationStatus.TERMINATE_ALL)
            logger.info(self.monitor_process_future.result())
            sleep(1)  # monitor が terminated 状態で少なくとも一度更新されなければ running のまま固まる

            # 全ての Exception を再表示
            for i, opt_exception in enumerate(opt_exceptions):
                if opt_exception is not None:
                    print()
                    print(f'===== unexpected exception raised on worker {i} =====')
                    print_exception(opt_exception)
                    print()

            # monitor worker を残してユーザーが結果を確認できるようにする
            if confirm_before_exit:
                print()
                print('='*len(Msg.CONFIRM_BEFORE_EXIT))
                print(Msg.CONFIRM_BEFORE_EXIT)
                print('='*len(Msg.CONFIRM_BEFORE_EXIT))
                input()

            return self.history.get_df()  # with 文を抜けると actor は消えるが .copy() はこの段階では不要

    @staticmethod
    def terminate_all():
        """Deprecated method. We plan to remove this in future version.

        In current version, the termination processes are
        automatically execute in the last of :func:`FEMOpt.optimize`.
        """
        warnings.warn(
            "terminate_all() is deprecated and will be removed in a future version. "
            "In current and later versions, the equivalent of terminate_all() will be executed when optimize() finishes. "
            "Therefore, you can simply remove terminate_all() from your code. "
            "If you want to stop program before terminating monitor process, "
            "use ``confirm_before_exit`` argument like ``FEMOpt.optimize(confirm_before_exit=True)``",
            DeprecationWarning,
            stacklevel=2
        )


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
