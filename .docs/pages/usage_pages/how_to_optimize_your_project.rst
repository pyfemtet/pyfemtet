最適化の実行手順
--------------------


このページでは、ご自身のプロジェクトに
pyfemtet.opt を適用して最適設計を行う際の
プログラム作成方法を示します。


1. Femtet プロジェクトの作成

    Femtet 上で解析モデルを作成します。
    **最適化したいパラメータを変数として登録してください。**
    パラメータを用いた解析設定の詳細については
    Femtet ヘルプ / プロジェクトの作成 / 変数 をご覧ください。


2. 目的関数の設定

    最適化問題では、評価したい指標を目的関数と呼びます。
    解析結果やモデル形状から目的関数を計算する処理を Femtet の Python マクロを用いて記述してください。

    .. code-block:: python

        """Femtet の解析結果から評価指標を計算する例です。"""
        from win32com.client import Dispatch

        # Femtet の操作のためのオブジェクトを取得
        Femtet = Dispatch("FemtetMacro.Femtet")

        # Femtet で解析結果を開く
        Femtet.OpenCurrentResult(True)
        Gogh = Femtet.Gogh

        # （例）応力解析結果からモデルの最大変位を取得
        dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()

    .. note::
        Femtet の Python マクロ文法は、Femtet マクロヘルプ又は
        `サンプルマクロ事例 <https://www.muratasoftware.com/support/macro/>`_
        をご覧ください。
    

3. メインスクリプトの作成

    上記で定義した設計変数と目的関数とを用い、メインスクリプトを作成します。

    .. code-block:: python

        """pyfemtet を用いてパラメータ最適化を行う最小限のコードの例です。"""

        from pyfemtet.opt import OptunaOptimizer

        def max_displacement(Femtet):
            """目的関数となる評価指標を計算する関数です。"""
            Gogh = Femtet.Gogh
            dx, dy, dz = Gogh.Galileo.GetMaxDisplacement()
            return dy
            
        if __name__ == '__main__':
            # 最適化を行うオブジェクトの準備
            femopt = OptunaOptimizer()

            # 設計変数の設定
            femopt.add_parameter('w', 10, 2, 20)
            femopt.add_parameter('d', 10, 2, 20)

            # 目的関数の設定
            femopt.add_objective(max_displacement, direction=0)

            # 最適化の実行
            femopt.optimize()

    .. note::
 
        目的関数は第一引数に Femtet インスタンスを取る必要があります。
        インスタンスは ~Optimizer クラス内で作成されるので、メインスクリプト内で定義しないでください。
        詳細は :doc:`../pages/examples` 又は :doc:`../pages/api` をご覧ください。 


    .. warning::
 
        ``add_parameter()`` は Femtet 内で定数式を設定した変数にのみ行い、
        文字式を設定した変数に対しては行わないでください。文字式が失われます。


4. スクリプトを実行します。

    スクリプトが実行されると、進捗および結果が csv ファイルに保存されます。
    csv ファイルの各行は一回の解析試行結果を示しています。各列の意味は以下の通りです。

    ===========  ======================================================
        列                                意味
    ===========  ======================================================
    trial        その試行が何度目の試行であるか
    <変数名>     スクリプトで指定した変数の値
    <目的名>     スクリプトで指定した目的関数の計算結果
    <目的名>     スクリプトで指定した目的関数の目標
    <拘束名>     スクリプトで指定した拘束関数の計算結果
    <拘束名>     スクリプトで指定した拘束関数の下限
    <拘束名>     スクリプトで指定した拘束関数の上限
    feasible     その試行がすべての拘束を満たすか
    hypervolume  （目的関数が2以上の場合のみ）その試行までのhypervolume
    message      最適化プロセスによる特記事項
    time         試行が完了した時刻
    ===========  ======================================================

    .. note:: <> で囲まれた項目はスクリプトに応じて内容と数が変化することを示しています。

    .. note:: 目的名、拘束名はスクリプト中で指定しない場合、obj_1, cns_1 などの値が自動で割り当てられます。
