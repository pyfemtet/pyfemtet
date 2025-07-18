from pathlib import Path
import webbrowser
import os
import urllib.parse

import pyfemtet
import subprocess
import pytest


project_root = Path(pyfemtet.__file__).parent.parent
docs_root = project_root / 'docs'
build_script = docs_root / 'pre-build.ps1'
ja_index_path = docs_root / 'build' / 'html_ja' / 'index.html'


@pytest.mark.manual
def test_check_documents():
    print('ドキュメントをビルドします。')
    print('主要ページの翻訳漏れがないこと、')
    print('主要 API の説明漏れがないことを確認します。')

    subprocess.run(
        [
            'powershell',
            '-ExecutionPolicy',
            'RemoteSigned',
            str(build_script)
        ],
    )
    
    # Windows用にスラッシュに変換
    try:
        abs_path = str(os.path.abspath(ja_index_path))
        abs_path = abs_path.replace('\\', '/')
        url = 'file:///' + urllib.parse.quote(abs_path)
        webbrowser.open(url)
    except Exception:
        print('Failed to open docs automatically...')
        print(abs_path)

    print('主要ページの翻訳漏れ又は')
    print('主要 API の説明漏れがあれば')
    print('NG と入力して Enter')
    result = input() or ''
    if result.upper() == 'NG':
        assert False, 'docs NG'
