from pyfemtet.opt.interface._base_interface import AbstractFEMInterface
from pyfemtet.opt.interface._femtet_interface import FemtetInterface
from pyfemtet.opt.interface._femtet_with_solidworks import FemtetWithSolidworksInterface
from pyfemtet.opt.interface._femtet_with_nx_interface import FemtetWithNXInterface

from pyfemtet.opt.problem.problem import TrialInput
from pyfemtet.beta.topology_matching import reexecute_model_with_topology_matching


class FemtetWithTopologyMatching(FemtetInterface, AbstractFEMInterface):

    def update_parameter(self, x: TrialInput, with_warning=False) -> None | list[str]:
        # Femtet のモデルを触らないようにする
        return AbstractFEMInterface.update_parameter(self, x)

    def update_model(self) -> None:
        # Femtet のモデルを触る処理はここに集約する
        def update():
            FemtetInterface.update_parameter(self, x=self.current_prm_values)
            FemtetInterface.update_model(self)

        reexecute_model_with_topology_matching(
            Femtet=self.Femtet,
            rebuild_fun=update,
        )


class FemtetWithSolidworksInterfaceWithTopologyMatching(
    FemtetWithSolidworksInterface, FemtetWithTopologyMatching
):
    # __init__ を FemtetWithSolidworksInterface から継承したいため
    # FemtetWithSolidworksInterface の update_parameter を呼ばないように
    # update_parameter をオーバーライドする
    def update_parameter(self, x: TrialInput, with_warning=False) -> None:
        return FemtetWithTopologyMatching.update_parameter(self, x)

    def update_model(self) -> None:
        # Femtet のモデルを触る処理はここに集約する
        def update():
            FemtetWithSolidworksInterface.update_parameter(self, x=self.current_prm_values)
            FemtetWithSolidworksInterface.update_model(self)

        reexecute_model_with_topology_matching(
            Femtet=self.Femtet,
            rebuild_fun=update,
        )



class FemtetWithNXInterfaceWithTopologyMatching(
    FemtetWithNXInterface, FemtetWithTopologyMatching
):
    def update_parameter(self, x: TrialInput, with_warning=False) -> None:
        return FemtetWithTopologyMatching.update_parameter(self, x)

    def update_model(self) -> None:
        # Femtet のモデルを触る処理はここに集約する
        def update():
            FemtetWithNXInterface.update_parameter(self, x=self.current_prm_values)
            FemtetWithNXInterface.update_model(self)

        reexecute_model_with_topology_matching(
            Femtet=self.Femtet,
            rebuild_fun=update,
        )
