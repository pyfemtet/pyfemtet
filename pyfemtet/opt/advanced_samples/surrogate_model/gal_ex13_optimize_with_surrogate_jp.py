import os

from optuna.samplers import TPESampler

from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface import PoFBoTorchInterface


def main(target):

    os.chdir(os.path.dirname(__file__))

    # Femtet との接続の代わりに、サロゲートモデルを作成します。
    # 学習データ作成スクリプトで作成した csv ファイルを読み込んで
    # サロゲートモデルを作成します。
    fem = PoFBoTorchInterface(
        history_path='training_data.csv'
    )

    # 最適化用オブジェクトの設定を行います。
    opt = OptunaOptimizer(
        sampler_class=TPESampler,
    )

    # FEMOpt オブジェクトの設定を行います。
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=f'optimized_result_target_{target}.csv'
    )

    # 設計変数の設定を行います。
    # 上下限は学習データ作成スクリプトと異なっても良いですが、
    # 学習していない範囲は外挿となりサロゲートモデルによる
    # 予測精度が低下することに注意してください。
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)

    # 学習時は設計変数としていたが最適化時に固定したいパラメータがある場合
    # initial_value のみを指定して fix 引数を True にしてください。
    # 学習時に設定しなかった設計変数を最適化時に追加することはできません。
    femopt.add_parameter('base_radius', 0.008, fix=True)

    # 学習時に設定した目的関数のうち
    # 最適化したいものを指定します。
    # fun 引数は与えてもいいですが、サロゲートモデル作成時に上書きされるため無視されます。
    # 学習時に設定しなかった目的関数を最適化時に使用することはできません。
    obj_name = '第一共振周波数(Hz)'
    femopt.add_objective(
        name=obj_name,
        direction=target,
    )

    # 最適化を実行します。
    femopt.set_random_seed(42)
    df = femopt.optimize(
        n_trials=50,
        confirm_before_exit=False
    )

    # 最適解を表示します。
    prm_names = femopt.history.prm_names
    obj_names = femopt.history.obj_names
    prm_values = df[df['non_domi'] == True][prm_names].values[0]
    obj_values = df[df['non_domi'] == True][obj_names].values[0]

    message = f'''
===== 最適化結果 =====    
ターゲット値: {target}
サロゲートモデルによる予測:
'''
    for name, value in zip(prm_names, prm_values):
        message += f'  {name}: {value}\n'
    for name, value in zip(obj_names, obj_values):
        message += f'  {name}: {value}\n'

    return message


if __name__ == '__main__':
    # 学習データから作成したサロゲートモデルで
    # 共振周波数が 1000 になる設計を見つけます。
    message_1000 = main(target=1000)

    # 続いて、同じサロゲートモデルで
    # 共振周波数が 2000 になる設計を見つけます。
    message_2000 = main(target=2000)

    print(message_1000)
    print(message_2000)
