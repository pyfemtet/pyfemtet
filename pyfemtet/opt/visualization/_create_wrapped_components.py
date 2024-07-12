"""Create autocompletable components"""
import os
import inspect

from dash.development.base_component import Component


# noinspection PyUnresolvedReferences
from dash import html, dcc, dash_table
# noinspection PyUnresolvedReferences
import dash_bootstrap_components as dbc


here, me = os.path.split(__file__)
COMPONENT_FILE_DIR = os.path.join(here, 'wrapped_components')
indent = '    '


def create(module_name: str) -> str:
    header = '''# auto created module
from pyfemtet.opt.visualization.wrapped_components.str_enum import StrEnum
# from enum import StrEnum
import dash
import dash_bootstrap_components


'''
    path = os.path.join(COMPONENT_FILE_DIR, module_name.replace('.py', '') + '.py')
    with open(path, 'w', newline='\n') as f:
        f.write(header)
    return path


def append(html_class, module_path: str):
    print('==========')
    print(html_class)

    # get property names
    init_signature = inspect.signature(html_class.__init__)
    props = [param.name for param in init_signature.parameters.values() if (param.name != 'self') and (param.name != 'args') and (param.name != 'kwargs')]

    # create class definition
    class_definition = f'class {html_class.__name__}({html_class.__module__}):\n'  # library specific

    # create id property
    class_definition += indent + 'def _dummy(self):\n'
    class_definition += indent*2 + '# noinspection PyAttributeOutsideInit\n'
    class_definition += indent*2 + 'self.id = None\n\n'

    # create Prop attribute
    class_definition += indent + 'class Prop(StrEnum):\n'
    for available_property in props:
        property_definition = f'{available_property} = "{available_property}"'
        try:
            exec(property_definition)
        except (SyntaxError, TypeError):
            continue
        class_definition += indent*2 + f'{property_definition}\n'

    if class_definition.endswith(':\n'):
        return

    class_definition += '\n\n'

    with open(module_path, 'a', newline='\n') as f:
        f.write(class_definition)


def sub(module_name):
    # glob component classes
    module_path = create(module_name)
    class_names = dir(eval(module_name))
    for class_name in class_names:
        cls = eval(f'{module_name}.{class_name}')
        if not inspect.isclass(cls):
            continue
        if issubclass(cls, Component):
            append(cls, module_path)


def main():
    # html
    sub('html')

    # dcc
    sub('dcc')

    # dbc
    sub('dbc')


if __name__ == '__main__':
    main()
