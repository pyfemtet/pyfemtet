from typing import TYPE_CHECKING
import platform

from ._base_interface import AbstractFEMInterface, NoFEM

# PyCharm 2025.1.2 では、以下のルールで参照先が決まる様子。仕様？
# if 文の前後で...
# - import されたクラスと定義されたクラスが混在する場合: 定義されたものが優先
# - import されたクラス同士である場合: 条件の True / False に関わらず、 else 内のものを優先
# - 定義されたクラス同士である場合: 条件の True / False に関わらず、 else 内のものを優先
# なのでわざわざ _dummy_classes module を分割作成し、条件式を反転して
# 実際の実装が参照されるようにした。
# VSCode では TYPE_CHECKING が True となる節が優先される。
if platform.system() != 'Windows' and not TYPE_CHECKING:
    from _dummy_classes import (
        FemtetInterface,
        FemtetWithNXInterface,
        FemtetWithSolidworksInterface,
        ExcelInterface,
    )

else:
    from ._femtet_interface import FemtetInterface
    from ._femtet_with_nx_interface import FemtetWithNXInterface
    from ._femtet_with_solidworks import FemtetWithSolidworksInterface
    from ._excel_interface import ExcelInterface
    from ._with_excel_settings import *
    from ._with_excel_settings import __all__ as _with_excel_settings__all__

from ._surrogate_model_interface import AbstractSurrogateModelInterfaceBase
from ._surrogate_model_interface import BoTorchInterface
from ._surrogate_model_interface import PoFBoTorchInterface

__all__ = [
    'AbstractFEMInterface',
    'NoFEM',
    'FemtetInterface',
    'FemtetWithNXInterface',
    'FemtetWithSolidworksInterface',
    'ExcelInterface',
    'AbstractSurrogateModelInterfaceBase',
    'BoTorchInterface',
    'PoFBoTorchInterface',
]

if platform.system() == 'Windows':
    __all__.extend(_with_excel_settings__all__)
