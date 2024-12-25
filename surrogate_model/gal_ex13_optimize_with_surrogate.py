import os

from optuna.samplers import TPESampler

from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface._singletaskgp import PoFBoTorchInterface


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))

    fem = PoFBoTorchInterface(
        history_path='241225_training_data.csv'
    )

    opt = OptunaOptimizer(
        sampler_class=TPESampler,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='241225_optimized_data.csv'
    )

    # 上下限は training と異なってもよいが、
    # 範囲を広げると外挿になるので注意
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)

    # training 時は振っていたが最適化時は
    # 固定したいパラメータは fix=True を使う。
    # 逆（training 時に振っていなかった変数を
    # 最適化時に追加する）はエラーになる
    femopt.add_parameter('base_radius', 0.008, fix=True)

    # training 時に指定した objective のうち
    # 最適化時に使用するものを記載します。
    # fun は無視されます（が、必須引数なので 空の関数 を渡す）
    obj_name = '0: 0番目のモード（共振） / 共振周波数[Hz]'
    femopt.add_objective(
        name=obj_name, fun=lambda: None,
        direction=1000,
    )

    femopt.optimize(
        n_trials=100,
    )
