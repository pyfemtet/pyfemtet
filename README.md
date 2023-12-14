# Welcome To PyFemtet !
pyfemtet.opt は、Femtet を用いてパラメータ最適化を行うことのできる Python パッケージです。

## 機能

Femtet を使ったシミュレーションによって、パラメータの最適化を行うことができます。
連続変数の単目的・多目的最適化に対応しています。
いくつかの最適化結果可視化機能を備えており、最適化結果の分析が可能です。

***注意：現在、本ライブラリは beta 版です！***

## Install

インストール方法は以下の通りです。

1. Femtet のインストール

    [https://www.muratasoftware.com/](https://www.muratasoftware.com/)

    初めての方は、試用版または個人版のご利用もご検討ください。


1. Femtet のマクロ有効化

    Femtet インストール後にスタートメニューから「マクロ機能を有効化する」を実行してください。

1. Python のインストール

    [https://www.python.org/](https://www.python.org/)

1. PyFemtet のインストール
    
    ターミナルで下記コマンドを実行してください。
    ```
    py -m pip install pyfemtet
    ```

1. Femtet マクロ定数の設定

    ターミナルで下記コマンドを実行してください。詳しくは Femtet マクロヘルプをご覧ください。
    ```
    py -m win32com.client.makepy FemtetMacro
    ```

    

## 動作するサンプルコード・サンプル解析モデル
pyfemtet プロジェクトの以下の相対パスに .py ファイルと、それと同名の .femprj ファイルが含まれています。
```
.\pyfemtet\FemtetPJTSample
```
このフォルダに含まれる ```.femprj``` ファイルは、変数設定済みの単純な解析モデルです。対応する同名の ```.py``` ファイルは、そのパラメータを PyFemtet を用いて最適化するサンプルコードです。 

Femtet でいずれかの ```.femprj``` ファイルを開き、その後対応する ```.py``` ファイルを実行してください。



## 使い方

基本的な使い方は以下の通りです。

1. Femtet プロジェクトの作成

    Femtet 上で解析モデルを作成します。最適化したいパラメータを変数として登録してください。

1. 評価指標の設定

    解析結果やモデル形状から評価したい指標を出力する処理を Femtet Python マクロを用いて記述してください。
    以下に例を示します。
    ```python
    """Femtet の解析結果から評価指標を計算する例です."""
    from win32com.client import constants
    from win32com.client import Dispatch

    # Femtet の操作のためのオブジェクトを取得
    Femtet = Dispatch("FemtetMacro.Femtet")

    # Femtet で解析結果を開く
    Femtet.OpenCurrentResult(True)
    Gogh = Femtet.Gogh

    # 流速を取得するための設定
    Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C

    # 目的の面で流速を積分し流量を取得
    _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)

    # 流量（次のステップでこれを評価指標にする）
    print(ret.Real)
    ```

1. メインスクリプトの作成

    上記で定義した評価指標を含むスクリプトを書いてください。以下に例を示します。前のステップで記述した評価指標が ```get_flow``` 関数の中に記述されています。

    ```python
    """pyfemtet を用いてパラメータ最適化を行う最小限のコードの例です.

    h, r という変数を有する解析モデルで簡易流体解析を行い, ある面の流量を 0.3 にしたい場合を想定しています.
    """

    from pyfemtet.opt import FemtetInterface, OptimizerOptuna
    from win32com.client import constants


    def get_flow(Femtet):
        """解析結果から流量を取得します.
        
        pyfemtet で定義する評価関数は、
        第一引数に Femtet の COM オブジェクトを
        設定する必要があります.
        """
        Gogh = Femtet.Gogh
        Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
        _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
        flow = ret.Real
        return flow


    if __name__ == '__main__':
   
        # Femtet 制御オブジェクトを用意
        fem = FemtetInterface('example.femprj')

        # 最適化処理を行うオブジェクトを用意
        femopt = OptimizerOptuna(fem)

        # 解析モデルで登録された変数
        femopt.add_parameter("h", 10, lower_bound=1, upper_bound=20, memo='高さ')
        femopt.add_parameter("r", 5, lower_bound=1, upper_bound=10, memo='半径')

        # 流量が 0.3 に近づくようにゴールを設定する
        femopt.add_objective(get_flow, name='流量', direction=0.3)

        # 最適化の実行（csv ファイルに最適化計算の過程が保存されます）
        femopt.main(n_trial=30)

        # 最適化結果の表示（最適化終了時点での csv ファイルの内容と同じです）
        print(femopt.history.data)
    ```
    注意：Femtet 内で数式を設定した変数に対し ```add_parameter()``` を行わないでください。数式が失われます。

1. スクリプトを実行します。
 
1. 出力された最適化過程の一覧を csv ファイルで確認します。

    csv ファイルの各行は一回の解析試行結果を示しています。各列の形式は以下の通りです。

    列名 | 説明
    --- | ---
    n_trial | 解析を行った回数
    [param] | スクリプト中で設定した変数。変数の数だけ列が増える。
    [objective] | スクリプト中で設定した評価関数。評価関数の数だけ列が増える。
    [constraint] | スクリプト中で設定した拘束関数。拘束関数の数だけ列が増える。
    non_domi | 非劣解であるかどうか。
    fit | 拘束関数を満たすかどうか。
    optimization_message | 最適化アルゴリズムによる注記。
    hypervolume | その試行時点でのハイパーボリューム。
    time | その試行が完了した日時。

    - 拘束：ある数式が任意の範囲内にあるようにすることです。具体的な記述方法は**動作するサンプルコード**も参考にしてください。
    - 非劣解：劣解とは、解集合の中で、その解よりも全ての評価指標が好ましい解が存在する解のことです。非劣解とは、劣解ではない解のことです。単一の評価指標のみを持つ最適化問題の場合、非劣解は最適解です。


## API reference
作成中です。


---

## English version of this document
We're sorry, this section is under constructing.

---
Copyright (C) 2023 Murata Manufacturing Co., Ltd. All Rights Reserved.

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.