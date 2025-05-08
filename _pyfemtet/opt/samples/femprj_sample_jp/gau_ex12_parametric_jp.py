"""パラメトリック解析出力設定を目的関数とする最適化

Femtet のパラメトリック解析の結果出力設定機能で出力される値を
最適化の目的関数として使用する方法をデモします。
この機能により、目的関数をコーディングすることなく
最適化を実施できます。


注意:

    この機能を使う際は、以下のことに注意してください。

    - パラメトリック解析のスイープテーブルが削除されます。
    - 複素数やベクトルを出力する出力設定は、第一の値のみが
      目的関数として使用されます。（複素数の場合は実数、
      ベクトル値の場合は X 成分など）


対応するプロジェクト: gau_ex12_parametric.femprj
"""

from pyfemtet.opt import FEMOpt, FemtetInterface


if __name__ == '__main__':

    # Femtet の設定を参照するため、Femtet と接続を
    # 行うためのオブジェクトを初期化します。
    fem = FemtetInterface()

    # パラメトリック解析の結果出力設定を目的関数にします。
    # number は Femtet パラメトリック解析ダイアログの
    # 結果出力設定タブのテーブルの番号で、direction は
    # その目的関数の目標です(FEMOpt.add_objective と同様)。

    # 相互インダクタンス
    fem.use_parametric_output_as_objective(number=1, direction=1.5e-7)

    # コイル中央の磁界の強さ
    fem.use_parametric_output_as_objective(number=2, direction='minimize')

    # 最適化用オブジェクトを初期化します。
    # さきほど初期化した fem を渡します。
    femopt = FEMOpt(fem=fem)

    # パラメータを設定します。
    femopt.add_parameter('in_radius', 10, 5, 10)
    femopt.add_parameter('out_radius', 20, 20, 25)

    # 最適化を実行します。
    femopt.set_random_seed(42)  # 乱数シードの固定
    femopt.optimize(n_trials=20)
