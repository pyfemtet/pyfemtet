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


def _(en_message: str, jp_message=None, **kwargs):
    if (jp_message is not None) and ('japanese' in LOC.lower()):
        return jp_message.format(**kwargs)

    else:
        return gettext(en_message).format(**kwargs)
