import os
import locale

LOC, ENCODING = locale.getlocale()


if __name__ == '__main__':
    # set target to create .pot
    # noinspection PyUnresolvedReferences
    from gettext import gettext as _

else:

    # get translation
    if 'japanese' in LOC.lower():
        from babel.support import Translations

        translations = Translations.load(
            os.path.join(os.path.dirname(__file__), 'locales'),
            locales='ja'
        )
        _ = translations.gettext

    else:
        def _(x):
            return x
