from PyFemtet.opt import FemtetScipy

def L(Femtet):
    # L の取得
    Gogh = Femtet.Gogh
    # コイル名の取得
    cName = Gogh.Gauss.GetCoilList()[0]
    # インダクタンスの取得
    return Gogh.Gauss.GetL(cName, cName)


# 最適化処理を行うオブジェクトを用意
FEMOpt = FemtetScipy()

# 解析モデルで登録された変数
FEMOpt.add_parameter("h", 3, 1.5, memo='1巻きピッチ')
FEMOpt.add_parameter("r", 5, 3, memo='コイル半径')
FEMOpt.add_parameter("n", 5, 1, 20, memo='コイル半径')

# インダクタンスが 0.44 uF に近づくようにゴールを設定する
FEMOpt.add_objective(
    L,
    name='自己インダクタンス',
    direction=4.4e-07
    )

# 最適化実行中にその収束状況を表示する(experimental)
FEMOpt.set_process_monitor()

# 最適化の実行 ※実行すると、csv ファイルでの最適化過程の保存が始まります。
FEMOpt.main()

# 最適化過程の一覧表示（最適化終了時点での csv ファイルの内容と同じです）
print(FEMOpt.history)
