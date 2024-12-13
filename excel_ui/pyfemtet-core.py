"""同梱する Excel ファイルから pyfemtet を呼び出す際の pyfemtet スクリプト。

Excel 側では下記の前提を守ること。
- 設計変数を記載するシートは表形式を維持すること。
- 目的関数を記載するシートは表形式を維持すること。
- 最適化の各繰り返しの際に呼ばれる関数は拡張子のない femprj ファイル名を引数に取ること
- 最適化の





"""


import os

from fire import Fire


def get_sampler_class(sampling_method):
    if sampling_method is None:
        # default
        from pyfemtet.opt.optimizer import PoFBoTorchSampler
        return PoFBoTorchSampler
    elif sampling_method == 'QMC':
        from optuna.samplers import QMCSampler
        return QMCSampler
    elif sampling_method == 'PoFBoTorch':
        from pyfemtet.opt.optimizer import PoFBoTorchSampler
        return PoFBoTorchSampler
    elif sampling_method == 'Random':
        from optuna.samplers import RandomSampler
        return RandomSampler
    elif sampling_method == 'NSGA2':
        from optuna.samplers import NSGAIISampler
        return NSGAIISampler
    elif sampling_method == 'TPE':
        from optuna.samplers import TPESampler
        return TPESampler
    elif sampling_method == 'BoTorch':
        from optuna_integration import BoTorchSampler
        return BoTorchSampler
    else:
        raise NotImplementedError(f'The method {sampling_method} is not implemented.')


def core(
        xlsm_path: str,
        csv_path: str or None,
        femprj_path: str,  # xlsm と同じフォルダに配置する前提。
        model_name: str or None,
        input_sheet_name: str,

        output_sheet_name: str or None,
        procedure_name: str or None,  # 引数に拡張子を除く femprj ファイル名を取るように実装すること
        setup_procedure_name: str or None,
        teardown_procedure_name: str or None,

        sampler_class: type('BaseSampler') or None,
        sampler_kwargs: dict or None,

        n_parallel: int,
        n_trials: int or None,
        timeout: float or None,
        seed: int or None,
):
    from pathlib import Path
    from optuna.samplers import BaseSampler
    from pyfemtet.opt import FEMOpt, OptunaOptimizer
    from pyfemtet.opt.interface._excel_interface import ExcelInterface

    prj_name = os.path.basename(femprj_path).removesuffix('.femprj')

    fem = ExcelInterface(
        input_xlsm_path=xlsm_path,
        input_sheet_name=input_sheet_name,
        output_xlsm_path=None,
        output_sheet_name=output_sheet_name,
        procedure_name=procedure_name,
        procedure_args=(prj_name,) if model_name is None else (prj_name, model_name,),
        connect_method='new',
        setup_procedure_name=setup_procedure_name,
        teardown_procedure_name=teardown_procedure_name,
        related_file_paths=[Path(femprj_path)],
        visible=False,
        interactive=True,
    )


    opt = OptunaOptimizer(
        sampler_class=sampler_class,
        sampler_kwargs=sampler_kwargs,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=csv_path,
    )

    if seed is not None:
        femopt.set_random_seed(42)

    femopt.optimize(
        n_trials=n_trials,
        n_parallel=n_parallel,
        timeout=timeout,
        confirm_before_exit=False,
    )


def main(
        xlsm_path: str,
        femprj_path: str,  # xlsm と同じフォルダに配置する前提。
        input_sheet_name: str,
        n_parallel: int,

        model_name: str = None,
        csv_path: str or None = None,
        output_sheet_name: str or None = None,
        procedure_name: str or None = None,  # 引数に拡張子を除く femprj ファイル名を取るように実装すること
        setup_procedure_name: str or None = None,
        teardown_procedure_name: str or None = None,

        algorithm: str or None = None,

        n_trials: int or None = None,
        timeout: float or None = None,
        seed: int or None = None,

        **algorithm_settings: dict,

):
    import sys
    import inspect
    from pyfemtet.logger import get_module_logger

    # ----- Fire memo -----
    # print(csv_path)  # 与えなければ None
    # print(algorithm_settings)  # 与えなければ {}, 与えれば {'n_startup_trials': 10} など
    # print(n_parallel, type(n_parallel))  # int か float に自動変換される
    # print(timeout, type(timeout))  # int か float に自動変換される

    # ----- check args -----
    logger = get_module_logger('opt.script', __name__)

    os.chdir(os.path.dirname(__file__))

    # xlsm_path
    xlsm_path = os.path.abspath(xlsm_path)
    if not os.path.exists(xlsm_path):
        logger.error(f'{xlsm_path} が見つかりませんでした。')
        input('終了するには Enter を押してください。')
        sys.exit(1)

    # femprj_path
    femprj_path = os.path.abspath(femprj_path)
    if not os.path.exists(femprj_path):
        logger.error(f'{femprj_path} が見つかりませんでした。')
        input('終了するには Enter を押してください。')
        sys.exit(1)

    # n_parallel
    try:
        n_parallel = int(n_parallel)
    except ValueError:
        logger.error(f'n_parallel ({n_parallel}) は自然数にできません。')
        input('終了するには Enter を押してください。')
        sys.exit(1)

    # csv_path
    csv_path = os.path.abspath(csv_path) if csv_path is not None else csv_path

    # n_trials
    if n_trials is not None:
        try:
            n_trials = int(n_trials)
        except ValueError:
            logger.error(f'n_trials ({n_trials}) は自然数にできません。')
            input('終了するには Enter を押してください。')
            sys.exit(1)

    # timeout
    if timeout is not None:
        try:
            timeout = float(timeout)
        except ValueError:
            logger.error(f'timeout ({timeout}) は数値にできません。')
            input('終了するには Enter を押してください。')
            sys.exit(1)

    # seed
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            logger.error(f'seed ({seed}) は自然数にできません。')
            input('終了するには Enter を押してください。')
            sys.exit(1)

    # sampler
    try:
        sampler_class = get_sampler_class(algorithm)
    except NotImplementedError:
        logger.error(f'algorithm ({algorithm}) は非対応です。')
        input('終了するには Enter を押してください。')
        sys.exit(1)

    # sampler_kwargs
    sampler_kwargs = algorithm_settings
    available_sampler_kwarg_keys = inspect.signature(sampler_class).parameters.keys()
    for given_key in sampler_kwargs.keys():
        if given_key not in available_sampler_kwarg_keys:
            print()
            print(sampler_class.__doc__)
            print()
            logger.error(f'algorithm_setting の項目 ({given_key}) は {sampler_class.__name__} に設定できません。詳しくは上記のドキュメントをご覧ください。')
            input('終了するには Enter を押してください。')
            sys.exit(1)

    core(
        xlsm_path,
        csv_path,
        femprj_path,  # xlsm と同じフォルダに配置する前提。
        model_name,
        input_sheet_name,

        output_sheet_name,
        procedure_name,  # 引数に拡張子を除く femprj ファイル名を取るように実装すること
        setup_procedure_name,
        teardown_procedure_name,

        sampler_class,
        sampler_kwargs,

        n_parallel,
        n_trials,
        timeout,
        seed,
    )


if __name__ == '__main__':
    Fire(main)
