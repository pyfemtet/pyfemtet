Welcome to PyFemtet's documentation!
====================================

.. |Femtet| raw:: html

    <a href="https://www.muratasoftware.com/" target="_blank">muratasoftware.com</a>

.. |Python| raw:: html

    <a href="https://www.python.org/" target="_blank">python.org</a>


概要
----------

**pyfemtet は ムラタソフトウェア製 CAE ソフト Femtet の拡張機能を提供します。**

- pyfemtet はオープンソースライブラリであり、無償かつ商用利用可能です。
- Femtet 本体の使用にはライセンスが必要です。pyfemtet は Femtet 本体のライセンスを一切変更しません。
- 評価のための試用版 Femtet は ムラタソフトウェア にお問い合わせください。

    - ➡ |Femtet|


pyfemtet の主要機能
----------------------------

pyfemtet は Femtet の Python マクロインターフェースを利用して機能を提供するライブラリです。
現在、 **pyfemtet の唯一の機能は設計パラメータの最適化** であり、pyfemtet.opt サブパッケージとして実装されています。

pyfemtet.opt による最適化機能は、以下の特徴を有します。

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


1. **Femtet（2023.1 以降）のインストール**
    
    初めての方は、試用版または個人版のご利用をご検討ください。➡ |Femtet|

    
2. **Femtet のマクロ有効化**

    Femtet インストール後にスタートメニューから
    「マクロ機能を有効化する」を実行してください。
    この手順には管理者権限が必要です。


3. **Python（3.9 以上）のインストール**

    リンク先のダウンロードリンクから
    インストーラをダウンロードし、実行してください。➡ |Python|

    .. note::

        pyfemtet は バージョン 3.9 ~ 3.12 の Python に対応していますが、
        最適化手法の一部は 3.12 に対応していないライブラリに依存しています。

    .. tip::

        実施例の最適化手法のすべてを使うには、バージョン 3.11 以下の Python が必要です。
        最新版でないバージョンの python をダウンロードするには
        下記のスクリーンショットを参考に
        ご自身の環境に応じたインストーラをダウンロードしてください。

    .. figure:: python_download.png

    .. figure:: python_3.11.png
        :scale: 50%

        このスクリーンショットでは、64 bit 版 windows 向け python 3.11.7 のインストーラへの
        リンクの場所の例を示しています。

    .. figure:: python_install.png

        インストーラ画面。


4. **pyfemtet のインストール**

    コマンドプロンプトで下記コマンドを実行してください。ライブラリのダウンロード及びインストールが始まります。::

        py -m pip install pyfemtet --no-warn-script-location 
    
    インストールが終了すると、"Successfully installed " の表示の後、コマンドプロンプトの制御が戻ります。

    .. figure:: pip_while_install.png

        インストール中

    .. figure:: pip_complete_install.png

        インストール終了後

    .. note::

        環境によりますが、インストールには 5 分程度を要します。

    .. note::

        インストール終了時に ``[notice] A new release of pip is available:`` などの表示がされることがありますが、
        エラーではなく、無視しても問題ありません。

5. **Femtet マクロ定数の設定**

    コマンドプロンプトで下記コマンドを実行してください。::

        py -m win32com.client.makepy FemtetMacro

    インストールが終了すると、コマンドプロンプトの制御が戻ります。

    .. figure:: complete_makepy.png

        makepy 終了後



以上で終了です。

.. tip::
    
    動作確認には、はじめに :doc:`/examples` のサンプルを閲覧いただくことをお勧めします。



目次
--------

.. toctree::
    :maxdepth: 2

    ホーム <self>
    examples
    usage
    api
    LICENSE
