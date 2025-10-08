at = '@'
AT = '_at_'
hyphen = '-'
HYPHEN = '_hyphen_'
dot = '.'
DOT = '_dot_'


def convert_symbols(name):
    return (
        name
        .replace(dot, DOT)
        .replace(at, AT)
        .replace(hyphen, HYPHEN)
    )
