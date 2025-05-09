from pyfemtet._i18n import _


class DeprecationError(Exception):
    pass


raise DeprecationError(_(
    en_message='Starting from version 1.0, the import '
               'method for the Error classes has changed. '
               'Please change imports such as '
               '`from pyfemtet.core import ModelError` '
               'to `from pyfemtet.opt.exceptions import ModelError`. '
               'For more details, please see https://pyfemtet.readthedocs.io/en/stable/pages/migration_to_v1.html.',
    jp_message='バージョン 1.0.0 からは、Error クラスのインポート方法が'
               '変わりました。`from pyfemtet.core import ModelError` '
               'などは、 `from pyfemtet.opt.exceptions import ModelError` '
               'としてください。'
               '詳しくは、https://pyfemtet.readthedocs.io/ja/stable/pages/migration_to_v1.html をご覧ください。',
))
