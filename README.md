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

    

## document

下記をご覧ください。
https://pyfemtet.github.io/pyfemtet/


---

## English version of this document
We're sorry, this section is under constructing.

---
Copyright (C) 2023 Murata Manufacturing Co., Ltd. All Rights Reserved.

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.