import ast
import inspect


__all__ = [
    '_is_access_gogh',
    '_is_access_femtet',
]


def _get_scope_indent(source: str) -> int:
    SPACES = [' ', '\t']
    indent = 0
    while True:
        if source[indent] not in SPACES:
            break
        else:
            indent += 1
    return indent


def _remove_indent(source: str, indent: int) -> str:  # returns source
    lines = source.splitlines()
    edited_lines = [l[indent:] for l in lines]
    edited_source = '\n'.join(edited_lines)
    return edited_source


def _check_access_femtet_objects(fun, target: str = 'Femtet'):

    # 関数fのソースコードを取得
    source = inspect.getsource(fun)

    # ソースコードを抽象構文木（AST）に変換
    try:
        # instanceメソッドなどの場合を想定してインデントを削除
        source = _remove_indent(source, _get_scope_indent(source))
        tree = ast.parse(source)

    except Exception:
        return False  # パースに失敗するからと言ってエラーにするまででもない

    # if function or staticmethod, 1st argument is Femtet. Find the name.
    varname_contains_femtet = ''  # invalid variable name
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_arguments: ast.arguments = node.args

            args: list[ast.arg] = all_arguments.args
            # args.extend(all_arguments.posonlyargs)  # 先にこっちを入れるべきかも

            target_arg = args[0]

            # if class method or instance method, 2nd argument is it.
            # In this implementation, we cannot detect the FunctionDef is
            # method or not because the part of source code is unindented and parsed.
            if target_arg.arg == 'self' or target_arg.arg == 'cls':
                if len(args) > 1:
                    target_arg = args[1]
                else:
                    target_arg = None

            if target_arg is not None:
                varname_contains_femtet = target_arg.arg

    # check Femtet access
    if target == 'Femtet':
        for node in ast.walk(tree):

            # by accessing argument directory
            if isinstance(node, ast.Name):
                # found local variables
                node: ast.Name
                if node.id == varname_contains_femtet:
                    # found Femtet
                    return True

            # by accessing inside method
            elif isinstance(node, ast.Attribute):
                # found attribute of something
                node: ast.Attribute
                if node.attr == 'Femtet':
                    # found **.Femtet.**
                    return True

    # check Gogh access
    elif target == 'Gogh':
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == 'Gogh':
                    # found **.Gogh.**
                    node: ast.Attribute
                    parent = node.value

                    # by accessing argument directory
                    if isinstance(parent, ast.Name):
                        # found *.Gogh.**
                        parent: ast.Name
                        if parent.id == varname_contains_femtet:
                            # found Femtet.Gogh.**
                            return True

                    # by accessing inside method
                    if isinstance(parent, ast.Attribute):
                        # found **.*.Gogh.**
                        parent: ast.Attribute
                        if parent.attr == 'Femtet':
                            # found **.Femtet.Gogh.**
                            return True

    # ここまで来たならば target へのアクセスはおそらくない
    return False


def _is_access_gogh(fun: callable) -> bool:
    return _check_access_femtet_objects(fun, target='Gogh')


def _is_access_femtet(fun: callable) -> bool:
    return _check_access_femtet_objects(fun, target='Femtet')
