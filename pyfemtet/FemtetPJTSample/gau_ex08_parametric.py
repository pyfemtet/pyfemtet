from pyfemtet.opt import OptimizerOptuna


def L(Femtet):
    # L の取得
    Gogh = Femtet.Gogh
    # コイル名の取得
    cName = Gogh.Gauss.GetCoilList()[0]
    # インダクタンスの取得
    return Gogh.Gauss.GetL(cName, cName)


if __name__ == '__main__':

    # 最適化処理を行うオブジェクトを用意
    femopt = OptimizerOptuna()

    # 解析モデルで登録された変数
    femopt.add_parameter("h", 3, 1.5, 6, memo='1巻きピッチ')
    femopt.add_parameter("r", 5, 3, 12, memo='コイル半径')
    femopt.add_parameter("n", 5, 1, 20, memo='コイル半径')

    # インダクタンスが 0.44 uF に近づくようにゴールを設定する
    femopt.add_objective(
        L,
        name='自己インダクタンス',
        direction=4.4e-07
        )

    # 最適化の実行
    femopt.main(n_trials=30, method='botorch', n_parallel=2)

    # プロセスモニターを終了
    femopt.terminate_monitor()
