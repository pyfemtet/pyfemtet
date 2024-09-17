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

from fire import Fire

import util

from_code = "en"
to_code = "ja"


def translate_to_japanese(string_to_translate):
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
        string_to_translate = string_to_translate.replace('\r\n', ' ')

    # 連続した空白を圧縮するかどうか
    if IGNORE_MULTIPLE_SPACE:
        string_to_translate = re.sub(r'[^\S\r\n]+', ' ', string_to_translate)

    # Translate
    translatedText = argostranslate.translate.translate(string_to_translate, from_code, to_code)

    # 句読点の後の空白を削除する
    translatedText = translatedText.replace('。 ', '。')

    # 行の文字数上限とインデントを処理
    if CHAR_PER_LINE > 0:
        # ===== インデントしつつ一行当たりの文字列を設定する ※ PlemolJP35 前提 =====

        # インデントと改行をすべて破棄した文字列を作成
        translatedText = ''.join([line.lstrip(' ') for line in translatedText.splitlines()])
        # インデントを含め x 文字までで分割する
        x = CHAR_PER_LINE - output_indent

        out = ' ' * output_indent  # 一行目のインデント
        current_char_per_line = output_indent
        for i, char in enumerate(translatedText):
            # 幅を調べる
            char_width = 1 if len(char) == len(char.encode('utf-8')) else 5 / 3
            # 現在の行に文字を追加し、その幅を計算する
            out += char
            current_char_per_line += char_width
            # 現在の行が上限に達したら改行を入れる
            if current_char_per_line > CHAR_PER_LINE:
                # ただし次の文字列が 。 とかなら改行しない
                try:
                    next_char = translatedText[i + 1]
                    if next_char in '。、"\'`_])}）':
                        continue
                except IndexError:
                    # 現在の文字が最後である場合。わざわざ改行しなくてよい。
                    break

                # ここまで来ていたら、通常の処理
                out += '\n'  # 改行
                out += ' ' * output_indent  # 次の行のインデント
                current_char_per_line = output_indent  # 行ごとの文字数の初期化


        translatedText = out

    else:
        if output_indent > 0:
            # ===== インデントのみする =====
            # 改行をリストに変換
            lines = translatedText.splitlines()
            # 既存のインデントを削除する
            lines = [line.lstrip(' ') for line in lines]
            # インデントをつける
            lines = [' ' * output_indent + line for line in lines]
            # リストを文字列に変換
            translatedText = '\n'.join(lines)
        else:
            # ===== 何もしない =====
            pass

    return translatedText


def main():
    # 翻訳プログラムの開始
    start = time()

    # クリップボードの文字列を取得
    string_to_translate = pyperclip.paste()

    translated_text = translate_to_japanese(string_to_translate)

    # 結果をクリップボードにコピー
    pyperclip.copy(translated_text)

    # 結果表示
    print('和訳にかかった時間: ', int(time() - start), '秒')
    print(' ===== 以下和訳（クリップボードにコピー済み） =====')
    print(translated_text)


if __name__ == '__main__':
    Fire(main)