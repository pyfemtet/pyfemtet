import os

from fire import Fire

from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface._excel_interface import ExcelInterface

here = os.path.dirname(__file__)
os.chdir(here)


def get_sampler(sampling_method):
    if sampling_method == 'QMC':
        from optuna.samplers import QMCSampler
        return QMCSampler()
    elif sampling_method == 'PoFBoTorch':
        from pyfemtet.opt.optimizer import PoFBoTorchSampler
        return PoFBoTorchSampler()
    elif sampling_method == 'Random':
        from optuna.samplers import RandomSampler
        return RandomSampler()
    elif sampling_method == 'NSGA2':
        from optuna.samplers import NSGAIISampler
        return NSGAIISampler()
    elif sampling_method == 'TPE':
        from optuna.samplers import TPESampler
        return TPESampler()
    elif sampling_method == 'BoTorch':
        from optuna_integration import BoTorchSampler
        return BoTorchSampler()
    else:
        raise NotImplementedError(f'The method {sampling_method} is not implemented.')


def core(
        femprj_path,
        input_xlsm_path,
        input_sheet_name,
        output_sheet_name,
        sampling_method,
        n_trials,
        timeout,
        seed,
        history_csv_path,
        n_parallel,
        model_name,
):

    fem = ExcelInterface(
        input_xlsm_path=input_xlsm_path,
        input_sheet_name=input_sheet_name,
        output_sheet_name=output_sheet_name,
        procedure_name='FemtetMacro.FemtetMain',
        procedure_args=None,
        procedure_timeout=timeout,
    )

    opt = OptunaOptimizer(
        sampler_class=get_sampler(sampling_method),
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=history_csv_path,
    )

    femopt.set_random_seed(seed)

    df = femopt.optimize(
        n_trials=n_trials,
        confirm_before_exit=True,
        n_parallel=n_parallel,
    )


def main(
        femprj_path: str,
        input_xlsm_path: str,
        input_sheet_name: str,
        output_sheet_name: str,
        sampling_method: str,
        n_trials: int or None = None,
        timeout: float or None = None,
        seed: int or None = None,
        history_csv_path: str or None = None,
        n_parallel: int = 1,
        model_name: str or None = None,
):
    """付属の Excel UI を使った場合のエントリポイントです。

    Args:
        femprj_path (str):
            FEMプロジェクトのファイルパスを指定します。

        input_xlsm_path (str):
            入力シートを含む excel ファイルパスを指定します。

        input_sheet_name (str):
            入力シートの名前を指定します。

        output_sheet_name (str):
            出力シートの名前を指定します。

        sampling_method (str):
            サンプリング方法を指定します。

        n_trials (int, optional):
            試行回数を指定します。デフォルトは
            Noneです。

        timeout (float, optional):
            タイムアウト時間を指定します。
            デフォルトはNoneです。

        seed (int, optional):
            乱数シードを指定します。デフォルトは
            Noneです。

        history_csv_path (str, optional):
            実行履歴を保存するCSVファイルの
            パスを指定します。デフォルトはNoneです。

        n_parallel (int, optional):
            並列処理の数を指定します。デフォルトは
            1です。

        model_name (str, optional):
            使用するモデルの名前を指定します。
            デフォルトはNoneです。


    """
    core(
        femprj_path=femprj_path,
        input_xlsm_path=input_xlsm_path,
        input_sheet_name=input_sheet_name,
        output_sheet_name=output_sheet_name,
        sampling_method=sampling_method,
        n_trials=n_trials,
        timeout=timeout,
        seed=seed,
        history_csv_path=history_csv_path,
        n_parallel=n_parallel,
        model_name=model_name,
    )


if __name__ == '__main__':
    try:
        Fire(main)
    except Exception as e:
        print(e)
        input()

