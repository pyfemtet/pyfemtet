# Welcome To PyFemtet.opt !
PyFemtet.opt は、Femtet を用いてパラメータ最適化を行うことのできる Python パッケージです。

## 機能

Femtet を使ったシミュレーションによって、パラメータの最適化を行うことができます。
連続変数の単目的・多目的最適化に対応しています。
いくつかの最適化結果可視化機能を備えており、最適化結果の分析が可能です。

***注意：現在、本ライブラリは α 版です！***

## install

インストール方法は以下の通りです。

1. Femtet のインストール

    [https://www.muratasoftware.com/](https://www.muratasoftware.com/)


1. Femtet のマクロ有効化

    詳しくは、Femtet インストール後にスタートメニューから「マクロ機能を有効化する」を選択してください。
    マクロ機能を利用するためには excel が必要です。

1. Python のインストール

    [https://www.python.org/](https://www.python.org/)

1. PyFemtet のインストール

    ```pip install PyFemtet```

1. Femtet のアンインストール
    PyFemtet をインストールした環境で下記のコマンドを実行してください。依存ライブラリは削除されません。
    ```
    pip uninstall PyFemtet
    ```

## 動作するサンプルコード・サンプル解析モデル
.zip ファイル内の以下の位置に、.py ファイルと、それと同名の .femprj ファイルが含まれています。
```
.\src\PyFemtet\FemtetPJTSample
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
    from win32com.client import constants
    from win32com.client import Dispatch
    Femtet = Dispatch("FemtetMacro.Femtet")

    # マクロから解析結果を開く
    Femtet.OpenCurrentResult(True)
    # 解析結果を取得するオブジェクトを作成
    Gogh = Femtet.Gogh
    # 流速を取得する設定
    Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
    # 目的の面で積分し流量を取得
    _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)

    print(ret.Real) # 流量（次のステップでこれを評価指標にする）
    ```

1. メインスクリプトの作成

    上記で定義した評価指標を含むスクリプトを書いてください。以下に例を示します。前のステップで記述した評価指標が ```get_flow``` 関数の中に記述されています。

    ```python
    from PyFemtet.opt import FemtetScipy
    from win32com.client import constants
    # from win32com.client import Dispatch
    # Femtet = Dispatch("FemtetMacro.Femtet") # このスクリプトでは使用しません

    '''h, r という変数を有する解析モデルで簡易流体解析を行い、ある面の流量を 0.3 にしたい場合を想定したスクリプト'''

    # 解析結果から流量を取得する関数
    def get_flow(Femtet):
        # この関数は、第一引数に Femtet のインスタンスを取るようにしてください。
        # Femtet.OpenCurrentResult(True) # この処理はあってもいいですが、不要です
        Gogh = Femtet.Gogh
        Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
        _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
        flow = ret.Real
        return flow

    # 最適化処理を行うオブジェクトを用意
    FEMOpt = FemtetScipy()

    # 解析モデルで登録された変数
    FEMOpt.add_parameter("h", 10, memo='高さ')
    FEMOpt.add_parameter("r", 5, memo='半径')

    # 流量が 0.3 に近づくようにゴールを設定する
    FEMOpt.add_objective(get_flow, name='流量', direction=0.3)

    # 最適化実行中にその収束状況を表示する(experimental)
    FEMOpt.set_process_monitor()

    # 最適化の実行 ※実行すると、csv ファイルでの最適化過程の保存が始まります。
    FEMOpt.main()

    # 最適化過程の一覧表示（最適化終了時点での csv ファイルの内容と同じです）
    print(FEMOpt.history)

    ```
    注意：Femtet 内で数式を設定した変数に対し ```add_parameter``` を行わないでください。数式が失われます。

1. ***Femtet で問題の解析モデルを開いた状態で***、スクリプトを実行します。

    - ***最適化を止めるには、計算中の Femtet を終了するか、スクリプトを実行している python プロセスを終了してください。*** それまでの最適化の過程は csv に保存されており、失われません。
    
1. 出力された最適化過程の一覧を確認します。一覧は表形式の構造を持っています。列は *変数、目的、拘束（設定した場合）、非劣解、拘束を満たすかどうか、解析終了時刻* です。各行は、その変数の組み合わせで FEM シミュレーションを行った結果の評価指標の値、拘束の値などを示しています。詳細は、以下も参考にしてください。
    - 拘束：ある数式が任意の範囲内にあるようにすることです。具体的な記述方法は**動作するサンプルコード**も参考にしてください。
    - 非劣解：劣解とは、解集合の中で、その解よりも全ての評価指標が好ましい解が存在する解のことです。非劣解とは、劣解ではない解のことです。単一の評価指標のみを持つ最適化問題の場合、非劣解は最適解です。


---

## English version of this document
We're sorry, this section is under constructing.

---
Copyright (C) 2023 kn514 <kazuma.naito@murata.com>  
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.