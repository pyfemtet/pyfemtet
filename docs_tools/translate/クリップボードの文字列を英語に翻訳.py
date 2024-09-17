"""クリップボードの文字列を日本語に翻訳します。"""

# ===== configuration =====
IGNORE_LINE_BREAK: bool = True
"""bool: 入力文字列中の改行を半角スペースに変換します。CHAR_PER_LINE が 0 以上の場合、True 扱いです。"""

IGNORE_MULTIPLE_SPACE: bool = True
"""bool: 入力文字列中の連続した半角スペースを 1 半角スペースに変換します。"""

CHAR_PER_LINE: int = 60
"""int: 出力文字列を一行あたり何文字にするかです。-1 のとき、改行しません。"""

OUTPUT_INDENT: int = -1
"""int: 出力文字列をいくつの半角スペースでインデントするかです。-1 のとき、入力文字列に合わせます。"""

# ===== main code =====
from time import time
import re

import pyperclip

import argostranslate.package
import argostranslate.translate

import util

from_code = "ja"
to_code = "en"


def translate_to_english(string_to_translate):
    # 翻訳パッケージのアプデ？
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    # 必要ならインデントを調べる
    if OUTPUT_INDENT < 0:
        output_indent = util.get_indent(string_to_translate)
    else:
        output_indent = OUTPUT_INDENT

    # 改行を無視するかどうか
    if IGNORE_LINE_BREAK:
        string_to_translate = string_to_translate.replace('\r\n', '')

    # 連続した空白を圧縮するかどうか
    if IGNORE_MULTIPLE_SPACE:
        string_to_translate = re.sub(r'[^\S\r\n]+', ' ', string_to_translate)

    # Translate
    translated_text = argostranslate.translate.translate(string_to_translate, from_code, to_code)

    # 行の文字数上限とインデントを処理
    if CHAR_PER_LINE > 0:
        # ===== インデントしつつ一行当たりの文字列を設定する ※ PlemolJP35 前提 =====

        # インデントと改行をすべて破棄した文字列を作成
        translated_text = ' '.join([line.lstrip(' ') for line in translated_text.splitlines()])
        # インデントを含め x 文字までで分割する
        x = CHAR_PER_LINE - output_indent

        out = ' ' * output_indent  # 一行目のインデント
        current_char_per_line = output_indent
        for char in translated_text:
            # 幅を調べる
            char_width = 1 if len(char) == len(char.encode('utf-8')) else 5 / 3

            # 現在の行に文字を追加したとすると、その幅を計算する
            out += char
            current_char_per_line += char_width

            # 現在の行が上限に達したら次の空白を改行に変換する
            if current_char_per_line > CHAR_PER_LINE:
                if char == ' ':
                    out = out[:-1]  # すでに足されている行末 ' ' を削除
                    out += '\n'  # 改行
                    out += ' ' * output_indent  # 次の行のインデント
                    current_char_per_line = output_indent  # 行ごとの文字数の初期化
        translated_text = out

    else:
        if output_indent > 0:
            # ===== インデントのみする =====
            # 改行をリストに変換
            lines = translated_text.splitlines()
            # 既存のインデントを削除する
            lines = [line.lstrip(' ') for line in lines]
            # インデントをつける
            lines = [' ' * output_indent + line for line in lines]
            # リストを文字列に変換
            translated_text = '\n'.join(lines)
        else:
            # ===== 何もしない =====
            pass

    return translated_text


if __name__ == '__main__':
    # 翻訳プログラムの開始
    start = time()

    # クリップボードの文字列を取得
    g_string_to_translate = pyperclip.paste()

    # 翻訳
    g_translatedText = translate_to_english(g_string_to_translate)

    # 結果をクリップボードにコピー
    pyperclip.copy(g_translatedText)

    # 結果表示
    print('英訳にかかった時間: ', int(time() - start), '秒')
    print(' ===== 以下英訳（クリップボードにコピー済み） =====')
    print(g_translatedText)