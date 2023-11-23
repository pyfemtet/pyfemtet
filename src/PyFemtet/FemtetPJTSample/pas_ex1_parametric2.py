# -*- coding: utf-8 -*-
'''
対応する解析モデル：pas_ex1_parametric.femprj
この解析モデルでは、円柱状の障害物を有する円筒管内に流体を流す解析を行います。
設計パラメータは以下の通りです。
r：障害物の半径
h：障害物の長さ
p：流体を流すための速度ポテンシャル
流量を特定の値にするために最適な r, h, p を求めます。
'''

from PyFemtet.opt import FemtetOptuna
from win32com.client import constants


#### 問題の設定

# 評価指標の定義（今回は流量）
def flow(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    Gogh = Femtet.Gogh
    Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
    _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
    flow = ret.Real
    return flow

# 拘束の定義（今回は障害物の体積）
def volume(Femtet):
    '''拘束を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    r = Femtet.GetVariableValue('r')
    h = Femtet.GetVariableValue('h')
    return 3.14 * r**2 * h


#### processing

# 最適化連携オブジェクトの準備
FEMOpt = FemtetOptuna() # ここで起動している Femtet が紐づけされます

#### 最適化の設定
FEMOpt.add_parameter('r', 50, lower_bound=0, upper_bound=99)
FEMOpt.add_parameter('h', 200, lower_bound=0, upper_bound=290)

FEMOpt.add_objective(flow, direction='maximize', name='流量')
FEMOpt.add_objective(volume, name='体積')

FEMOpt.add_constraint(volume, name='障害物の体積',
                      lower_bound=0, upper_bound=3.14 * 20**2 * 150)

# 進行状況の表示
# 一度画面を閉じると再表示できません。
FEMOpt.set_process_monitor()

# 最適化の実行
opt = FEMOpt.main()


#### 結果表示
# 実行時間表示
print(f'実行時間は約 {int(FEMOpt.lastExecutionTime)//60} 分でした。')
# 結果履歴表示
print(FEMOpt.history)






