import os
import locale

__all__ = [
    'LOC', 'ENCODING', '_'
]

LOC, ENCODING = locale.getlocale()


if __name__ == '__main__':
    # set target to create .pot
    # noinspection PyUnresolvedReferences
    from gettext import gettext

else:
    # get translation
    if 'japanese' in LOC.lower():
        from babel.support import Translations

        translations = Translations.load(
            os.path.join(os.path.dirname(__file__), 'locales'),
            locales='ja'
        )
        gettext = translations.gettext

    else:
        def gettext(x):
            return x


def _(message: str, **kwargs):
    return gettext(message).format(**kwargs)
