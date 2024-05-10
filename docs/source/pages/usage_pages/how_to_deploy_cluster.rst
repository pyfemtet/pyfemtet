.. |dask| raw:: html

    <a href="https://docs.dask.org/en/stable/deploying.html" target="_blank">dask documentation</a>


（実験的機能）クラスタ計算の実行手順
--------------------------------------------


このページでは、pyfemtet.opt を用いた最適化プログラムを
複数台の PC を用いて並列計算する際の手順を示します。

.. note::
    
    ここでは、 **プログラムを呼び出す手元マシンを「手元 PC」、計算を実行するマシンを「計算 PC」と呼びます。**
    計算 PC は複数あっても構いません。手元マシンが計算マシンであっても構いません。
    計算 PC 1 台ごとに「計算 PC のセットアップ」および「Worker の起動」を行ってください。


.. tip::
    
    pyfemtet の並列計算は ``dask.distributed`` に依存しています。
    また本ドキュメントは dask version 2023.12.1 時点での挙動を説明しています。
    詳細、および最新の CLI コマンド使用方法は |dask| をご覧ください。


1. プログラムの作成

    :doc:`how_to_optimize_your_project`  などを参考に、
    最適化を行うプログラムを作成してください。


2. 計算 PC のセットアップ

    - 計算 PC に Femtet をインストールしてください。
    - 計算 PC に手元 PC と同じバージョンの Python をインストールしてください。
    - 計算 PC に手元 PC と同じバージョンの pyfemtet および依存ライブラリをインストールしてください。

        - 依存ライブラリのバージョンを指定してインストールするには、下記手順が便利です。コマンドプロンプトから下記手順を実行してください。
          # 以降はコメントなので、実行しないでください。

        .. code-block::

            # 手元 PC
            # カレントディレクトリに requirements.txt というファイルを作成し
            # 現在の環境にインストールされているライブラリ一覧を
            # バージョンとともに書き出すコマンドです。
            pip freeze > requirements.txt

        ここで生成された requirements.txt というファイルを計算 PC に転送し、
        コマンドプロンプトで下記コマンドを実行します。

        .. code-block::
            
            # 計算 PC
            # requirements.txt を読み込み、記載通りのバージョンのライブラリを
            # インストールするコマンドです。
            pip install -r requirements.txtのパス

        makepy コマンドを実行し、Femtet のマクロ定数の設定を行ってください。

        .. code-block::
            
            # 計算 PC
            py -m win32com.client.makepy FemtetMacro


3. Scheduler（複数の PC のプロセスを管理するプロセス）の起動

    - 手元 PC で下記コマンドを実行してください。

        .. code-block::

            # 手元 PC
            # このコマンドの実行後、コマンドプロンプトは
            # 通信待ち状態となり制御を受け付けなくなります。
            # 終了時は Ctrl+C で終了してください。
            dask scheduler 

        .. figure:: images/dask_scheduler.png

            ここで表示される tcp://~~~:~~~ という数字を記録してください。

        .. note::

            | ファイアウォール等の制約により通信できるポートが決まっている場合は、
            | ``dask scheduler --port your_port``
            | コマンドを使用してください（your_port はポート番号に置き換えてください）。


4. Worker（計算を実行するプロセス）の起動

    - 計算 PC で下記コマンドを実行してください。

        .. code-block::

            # 計算 PC
            # このコマンドの実行後、コマンドプロンプトは
            # 通信待ち状態となり制御を受け付けなくなります。
            # 終了時は Ctrl+C で終了してください。
            dask worker tcp://~~~:~~~ --nthreads 1 --nworkers -1

        scheduler, worker 双方で画面が更新され、
        ``Starting established connection`` という
        文字が表示されれば通信が成功しています。

        .. note:: 通信できない状態で一定時間が経過すると、Worker 側でタイムアウトした旨のメッセージが表示されます。
        

5. プログラムの編集と実行

    - プログラムに Scheduler のアドレスを記載し、プログラム実行時に Scheduler に計算タスクが渡されるようにします。
    - FEMOpt コンストラクタの引数 ``scheduler_address`` に ``tcp://~~~:~~~`` を指定してください。

        .. code-block:: Python

            from pyfemtet.opt import FEMOpt

            ...  # 目的関数の定義など

            if __name__ == '__main__':

                ...  # fem, opt のセットアップなど

                femopt = FEMOpt(scheduler_address='tcp://~~~:~~~')

                ...  # 最適化問題のセットアップなど
        
                femopt.main()  # クラスターに接続され、最適化が実行されます。

                # femopt.terminate_all()  # 手順 3, 4 で起動した Scheduler, Worker 等のプロセスを自動で終了します。


.. warning::

    エラー等でプログラムが異常終了した場合、再試行の前に Scheduler, Worker を一度終了し、
    もう一度手順 3, 4 を実行することをお勧めします。



