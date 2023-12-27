# 使い方

## インストール

pyfemtet は windows OS にのみ対応しています。
インストール方法は以下の通りです。

1. Femtet のインストール

    [https://www.muratasoftware.com/](https://www.muratasoftware.com/)

    初めての方は、試用版または個人版のご利用をご検討ください。


1. Femtet（>=2023.1） のマクロ有効化

    Femtet インストール後にスタートメニューから「マクロ機能を有効化する」を実行してください。


1. Python（>=3.11） のインストール

    [https://www.python.org/](https://www.python.org/)


1. pyfemtet のインストール
    
    ターミナルで下記コマンドを実行してください。
    ```
    py -m pip install pyfemtet
    ```


1. Femtet マクロ定数の設定

    ターミナルで下記コマンドを実行してください。
    ```
    py -m win32com.client.makepy FemtetMacro
    ```


## pyfemtet を用いた最適化の実行手順

基本的な使い方は以下の通りです。

1. __Femtet プロジェクトの作成__

    Femtet 上で解析モデルを作成します。最適化したいパラメータを変数として登録してください。

1. __評価指標の設定__

   解析結果やモデル形状から評価したい指標を出力する処理を Femtet Python マクロを用いて記述してください。
   以下に例を示します。
   ```python
   """Femtet の解析結果から評価指標を計算する例です。"""
   from win32com.client import Dispatch
   
   # Femtet の操作のためのオブジェクトを取得
   Femtet = Dispatch("FemtetMacro.Femtet")
   
   # Femtet で解析結果を開く
   Femtet.OpenCurrentResult(True)
   Gogh = Femtet.Gogh
   
   # （例）応力解析結果からモデルの最大変位を取得
   dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()
   ```


1. __メインスクリプトの作成__

   上記で定義した評価指標を含むスクリプトを書いてください。以下に例を示します。
   
   ```python
   """pyfemtet を用いてパラメータ最適化を行う最小限のコードの例です。"""
   
   from pyfemtet.opt import OptimizerOptuna
   
   def max_displacement(Femtet):
       Gogh = Femtet.Gogh
       dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()
       return dy
      
   if __name__ == '__main__':
       # 最適化を行うオブジェクトの準備
       femopt = OptimizerOptuna()

       # 設計変数の設定
       femopt.add_parameter('w', 10, 2, 20)
       femopt.add_parameter('d', 10, 2, 20)

       # 目的関数の設定
       femopt.add_objective(max_displacement, direction=0)

       # 最適化の実行
       femopt.main()
   ```

   注意：Femtet 内で数式を設定した変数に対し ```add_parameter()``` を行わないでください。数式が失われます。


1. __スクリプトを実行します。__

   スクリプトが実行されると、進捗および結果が csv ファイルに保存されます。
   csv ファイルの各行は一回の解析試行結果を示しています。各列の意味は以下の通りです。

| 列                 | 意味                                   |
|-------------------|--------------------------------------|
| trial             | その試行が何度目の試行であるか                      |
| <変数名>             | スクリプトで指定した変数の値                       |
| <目的関数名>           | スクリプトで指定した目的関数の計算結果                  |
| <目的関数名>_direction | スクリプトで指定した目的関数の目標                    |
| <拘束関数名>           | スクリプトで指定した拘束関数の計算結果                  |
| <拘束関数名>_lb        | スクリプトで指定した拘束関数の下限                    |
| <拘束関数名>_ub        | スクリプトで指定した拘束関数の上限                    |
| feasible          | その試行がすべての拘束を満たすか                     |
| hypervolume       | （目的関数が 2 以上の場合のみ）その試行までの hypervolume |
| message           | 最適化プロセスによる特記事項                       |
| time              | 試行が完了した時刻                            |

<> で囲まれた文字はスクリプトでに応じて内容と数が変化することを示しています。
