from pyfemtet.opt.interface._base_interface import AbstractFEMInterface
from pyfemtet.opt.interface._femtet_interface import FemtetInterface
from pyfemtet.opt.problem.problem import TrialInput
from pyfemtet.beta.topology_matching import reexecute_model_with_topology_matching


class FemtetWithTopologyMatching(FemtetInterface, AbstractFEMInterface):

    def update_parameter(self, x: TrialInput, with_warning=False) -> None | list[str]:
        # Femtet のモデルを触らないようにする
        return AbstractFEMInterface.update_parameter(self, x)

    def update_model(self) -> None:
        # Femtet のモデルを触る処理はここに集約する
        # withNX などとの Mixin を考慮して super() で呼ぶ
        s = super()
        def update():
            s.update_parameter(x=self.current_prm_values)
            s.update_model()

        reexecute_model_with_topology_matching(
            Femtet=self.Femtet,
            rebuild_fun=update,
        )
