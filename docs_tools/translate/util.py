def get_indent(string: str, bl: str = None):
    """入力文字列からインデントを推定します"""
    # 改行のデフォルトは \r\n (win の pyperclip copy が前提)
    bl = bl or '\r\n'

    # 入力文字列を bl で split し空白でないものを抽出
    lines = list(filter(lambda x: x, string.split(bl)))

    # 最初空白がいくつあるか調べる
    indent = 0
    line = lines[0]
    for char in line:
        if char != ' ':
            break
        indent += 1
    return indent


def test_get_indent():
    indent = get_indent("\r\n    aaa\r\n    bbb\r\n\r\n")
    assert indent == 4