from pathlib import Path
import pyfemtet
import subprocess
import pytest


project_root = Path(pyfemtet.__file__).parent.parent
docs_root = project_root / 'docs'
build_script = docs_root / 'pre-build.ps1'


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

    print('主要ページの翻訳漏れ又は')
    print('主要 API の説明漏れがあれば')
    print('NG と入力して Enter')
    result = input() or ''
    if result.upper() == 'NG':
        assert False, 'docs NG'
