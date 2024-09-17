""".po ファイルを読み込み、git 履歴をもとに変更された msgstr に対し翻訳を提案します。"""

import subprocess
import re

import argostranslate.package
import argostranslate.translate

file_path = './docs/source/locale/ja_JP/LC_MESSAGES/examples.po'
# file_path = './docs/source/locale/ja_JP/LC_MESSAGES/index.po'
# file_path = './docs/source/locale/ja_JP/LC_MESSAGES/pages.po'

from_code = 'en'
to_code = 'ja'

MO_ENCODING = 'utf-8'
""".po ファイルのエンコーディングです。"""

# 翻訳パッケージのアプデ？
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())


def get_modified_msgids(file_path):
    """変更または追加された msgid とその行数を返します。"""

    # git diff で差分を抽出
    diff = subprocess.run(
        ['git', 'diff', '-U0', file_path],
        stdout=subprocess.PIPE,
    )
    diff_lines = diff.stdout.decode().splitlines()

    # 差分の中から msgid で始まる文字列を抽出
    msgids = []
    msgid_pattern = re.compile(r'^\+msgid\s"(.*)"$')
    for line in diff_lines:
        match = msgid_pattern.match(line)
        if match:
            msgids.append(match.group(1))

    # msgid を含む行の行インデックスをマップ
    with open(file_path, 'r', encoding=MO_ENCODING) as f:
        content: list[str] = f.readlines()

    line_nums = []
    for msgid in msgids:
        for i, line in enumerate(content):
            if msgid in line:
                line_nums.append(i)
                break

    return msgids, line_nums


def gen_translation(msgid):
    # Translate
    translatedText = argostranslate.translate.translate(msgid, from_code, to_code)

    return translatedText


def update_po_file(file_path, msgid_translations, line_nums):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    in_msgid = False
    current_msgid = None

    for line_num, translated in zip(line_nums, msgid_translations):
        lines[line_num + 1] = f'msgstr "{translated}"\n'

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def main():
    # ファイルパスを取得
    # file_path = ...

    # 変更または追加された msgid (原文) を抽出
    msgids, line_nums = get_modified_msgids(file_path)

    # 翻訳を追加
    translateds = []
    for msgid in msgids:
        translation = input(f'Translation for "{msgid}": {gen_translation(msgid)} or: ') or gen_translation(msgid)
        translateds.append(translation)

    update_po_file(file_path, translateds, line_nums)
    print("Translations updated.")


if __name__ == "__main__":
    main()
