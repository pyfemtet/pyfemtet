Welcome to PyFemtet's documentation!
====================================

pyfemtet は有限要素法解析ソフト `Femtet <https://www.muratasoftware.com/>`_ の Python インターフェースのパッケージです。

**pyfemtet は BSD-3 ライセンスの下で公開されているオープンソースライブラリです。
pyfemtet は Femtet 本体の機能およびライセンスを変更することなく
Femtet の商用利用可能な拡張機能を無償で提供します。
詳しくは** :doc:`LICENSE` **をご確認ください。**

現在の pyfemtet の唯一の主要なサブパッケージ pyfemtet.opt は
設計のパラメータ最適化を行うシンプルな API を提供します。


pyfemtet.opt の主要機能
----------------------------
- 単目的および多目的の最適化
- プロセスモニタによるリアルタイム進行状況の表示
- 複数の Femtet インスタンスによる並列計算
- Excel 等で分析が容易な csv 形式での結果出力


実施例
--------

.. grid:: 2

    .. grid-item-card:: ソレノイドコイルのインダクタンス
        :link: gau_ex08/gau_ex08
        :link-type: doc
        :text-align: center

        .. image:: gau_ex08/gau_ex08.png
            :scale: 50
        +++
        磁場解析で有限長ソレノイドコイルの自己インダクタンスを特定の値にします。


    .. grid-item-card:: 円形パッチアンテナの共振周波数
        :link: her_ex40/her_ex40
        :link-type: doc
        :text-align: center

        .. image:: her_ex40/her_ex40.png
            :scale: 50
        +++
        電磁波解析で円形パッチアンテナの共振周波数を特定の値にします。


.. tip::
    
    :doc:`examples` セクションにより多くの実施例があります。


シンプルな API
----------------------------

下記は多目的最適化の実施例です。使用方法の詳細は :doc:`usage` セクションを確認してください。

.. code-block:: python

   from pyfemtet.opt import OptimizerOptuna

   def max_displacement(Femtet):
       dx, dy, dz = Femtet.Gogh.Galileo.GetMaxDisplacement()
       return dy

   def volume(Femtet):
       w = Femtet.GetVariableValue('w')
       d = Femtet.GetVariableValue('d')
       h = Femtet.GetVariableValue('h')
       return w * d * h

   if __name__ == '__main__':
       femopt = OptimizerOptuna()
       femopt.add_parameter('w', 10, 2, 20)
       femopt.add_parameter('d', 10, 2, 20)
       femopt.add_objective(max_displacement, name='最大変位', direction=0)
       femopt.add_objective(volume, name='体積', direction='minimize')
       femopt.main(n_trials=20)

.. note::

   このサンプルを実施するためには :download:`サンプル プロジェクト <files/opt_sample.femprj>` をダウンロードし、Femtet で開いた状態で上記スクリプトを実行してください。


インストール
---------------

.. note:: pyfemtet は windows にのみ対応しています。

.. note::
    
    Python 及び Femtet がインストールされ
    Femtet のマクロが有効化されている環境では
    単に ``pip install pyfemtet`` を実行してください。
    以下の手順は、Python 及び Femtet のフルセットアップの手順です。


1. `Femtet（2023.1 以降）のインストール <https://www.muratasoftware.com/>`_
    
    初めての方は、試用版または個人版のご利用をご検討ください。

    
2. Femtet のマクロ有効化

    Femtet インストール後にスタートメニューから
    「マクロ機能を有効化する」を実行してください。
    この手順には管理者権限が必要です。


3. `Python（3.11）のインストール <https://www.python.org/>`_

    リンク先のダウンロード表示から **バージョン 3.11** の
    インストーラをダウンロードし、実行してください。

    .. note::

        pyfemtet は **python3.11 にのみ対応** しています。

    .. tip::
        手順 4 以降の py コマンドを実行するには、
        インストールウィザードにて "pip" 及び "py launcher" に
        チェックがあることを確認してください。
    
    .. figure:: py_launcher.png


4. pyfemtet のインストール

    コマンドプロンプトで下記コマンドを実行してください。ライブラリのダウンロード及びインストールが始まります。::

        py -m pip install pyfemtet


5. Femtet マクロ定数の設定

    コマンドプロンプトで下記コマンドを実行してください。::

        py -m win32com.client.makepy FemtetMacro


以上で終了です。動作確認には、はじめに :doc:`/examples` のサンプルを閲覧いただくことをお勧めします。


目次
--------

.. toctree::
    :maxdepth: 2

    ホーム <self>
    examples
    usage
    api
    LICENSE
