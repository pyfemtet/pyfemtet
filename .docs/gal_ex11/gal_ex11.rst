鐘の共振周波数
=====

Femtet の応力解析ソルバーを用い、鐘の第一共振周波数を
特定の値にする例題を解説します。


サンプルファイル
-----
.. note::

   :download:`サンプルプロジェクト<../../pyfemtet/FemtetPJTSample/gal_ex11_parametric.femprj>`
   を Femtet で開き、
   :download:`サンプルコード<../../pyfemtet/FemtetPJTSample/gal_ex11_parametric.py>`
   を実行してください。

.. note::

FEM 問題としての詳細については、FemtetHelp / 例題集 / 応力解析 / 例題11 を参照してください。


設計変数
-----

.. figure:: gau_ex11_model.png
   
   モデルの外観

====== ======
変数名 説明
====== ======
internal_r      鐘の外形
external_r      鐘の内径
h      鐘の円筒部の高さ
====== ======


目的関数
-----

自己インダクタンス


サンプルコード
-----

.. literalinclude:: ../../pyfemtet/FemtetPJTSample/gal_ex11_parametric.py
   :language: python
   :linenos:
   :caption: gal_ex11_parametric.py


サンプルコードの実行結果
-----

.. figure:: gal_ex11_result.png
   :width: 300

   gal_ex11_parametric.py の実行結果。
   横軸が試行回数、縦軸が第一共振周波数です。

30 回目の試行の結果、共振周波数は  Hz となります。

.. note::

   Femtet, pyfemtet および依存する最適化エンジンのバージョンにより、結果は多少異なる場合があります。
