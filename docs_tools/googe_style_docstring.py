# -*- coding: utf-8 -*-
"""Example Google style docstrings.


このモジュールは、`Google Python Style Guide`_ で指定されたドキュメ
ントを示します。 Docstring は複数の行を継承する場合があります。 セクシ
ョンヘッダとコロンは、インデントされたテキストのブロックに従って作成されま
す。

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    たとえば ``Example`` または ``Examples`` のセクションで指定
    できます。 セクションは、リテラルブロックを含む reStructuredText
    フォーマットをサポートします。

    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

セクション区切りは、インデントされていないテキストを再開することによって作
成されます。 セクション区切りはまた、新しいセクションが始まるたびに暗黙的に作成
されます。

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.


Attributes:
    module_level_variable1 (int): モジュールレベルの変数は、モジュールの
        docstring の ``Attributes`` セクション、または変数の直後に
        インラインの docstring のいずれかで文書化できます。

        Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        どちらのフォームも受け付けますが、2つは混在しないでください。 モ
        ジュールレベルの変数を文書化し、それに一貫してある慣習を選んで下
        さい。

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

module_level_variable1 = 12345

module_level_variable2 = 98765
"""int: インラインで文書化されたモジュールレベルの変数。

docstring は複数の行をスパンさせることができます。 型はコロンで区切る最
初の行で任意に指定することができます。

The docstring may span multiple lines. The type may optionally be specified
on the first line, separated by a colon.
"""


def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    `PEP 484`_ 型アノテーションがサポートされています。属性、パラメータ、
    および戻りタイプが `PEP 484`_ に従ってアノテートされている場合は、
    docstring に含まれる必要はありません。

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """


def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """


def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    関数パラメータは ``Args`` セクションで文書化されるべきです。各パラメ
    ータの名前が必要です。各パラメータの型と記述はオプションですが、明ら
    かでないならば記述すべきです。

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    \*args または \*\*kwargs を引数とする場合は、docstring で
    ``*args`` と ``**kwargs`` としてリストする必要があります。

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description

            説明は複数の行にスパンする場合があります。次の行はインデント
            する必要があります。"(type)" はオプションです。

            パラメータの説明で複数の段落がサポートされています。
            （訳注：空白行を挟めば段落扱いになるということ）

            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        戻りタイプはオプションで、 ``Returns`` セクションの先頭にコロ
        ンを追って指定できます。

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        ``Returns` セクションは、複数の行と段落に及ぶことがあります。
        次の行は、最初の行に一致するように刻まれるべきです。

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        ``Returns` セクションは、リテラルブロックを含むreStructured
        Textフォーマットをサポートしています。

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
            ``Raises` セクションは、インターフェイスに関連する
            すべての例外のリストです。
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True


def example_generator(n):
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n (int): The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        int: The next number in the range of 0 to `n` - 1.

    Examples:

        例はdoctest形式で記述し、関数の使い方を記述する必要があります。

        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        >>> print([i for i in example_generator(4)])  # doctest: +SKIP
        [0, 1, 2, 3]

    """
    for i in range(n):
        yield i


class ExampleError(Exception):
    """例外はクラスと同じ方法で文書化されます。

    __init_ メソッドは、クラスレベルの docstring か、 __init__ メソ
    ッド自体の docstring として記述できます。どちらのフォームも受け付け
    ますが、2つは混在しないでください。__init__ メソッドを文書化するには
    どちらか一方を選び、一貫してその方法を使用してください。

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        ``Args`` セクションに `self` パラメータを含まないでください。
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code


class ExampleClass(object):
    """クラス docstring の要約行は 1 行に収まる必要があります。

    クラスがパブリック属性を持っている場合は、関数の ``Args`` セク
    ションと同じ書式にしたがって``Attributes`` セクションで文書化
    できます。または、属性は属性の宣言でインラインに文書化される場合があり
    ます(下にある_init__メソッドを参照)。
    （訳注：宣言の後に #: で始まるコメント？）

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    ``@property`` デコレータで作成されたプロパティは、プロパティの getter
    メソッドで文書化する必要があります。

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """__init__メソッドのdocstringの例。

        __init_ メソッドは、クラスレベルの docstring か、 __init__
         メソッド自体の docstring として記述できます。どちらのフォーム
        も受け付けますが、2つは混在しないでください。__init__ メソッド
        を文書化し、それと一致させるための一つの規約を選択します。

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """:obj:`list` of :obj:`str`: getter と setter の両方の
        プロパティは、getter メソッドでのみ文書化されなくてはなりません。
        setter メソッドに注目すべき動作が含まれている場合、ここで説明する
        必要があります。

        Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """クラスメソッドの docstring は普通の関数に似ています。

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """デフォルトでは、docstring の特別なメンバーは（出力のドキュメントに）含まれません。

        By default, special members with docstrings are not included.

        特別なメンバーは、二重アンダースコアで始まり終了するメソッドま
        たは属性です。docstring を持つ特別なメンバーは、`napoleon_i
        nclude_special_with_doc` が True に設定されている場合に出
        力に含まれています。この動作は、Sphinx の conf.py で次の設定
        を変更することで有効にすることができます:

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """デフォルトではプライベートメンバーは（ドキュメントに）含まれません。

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass