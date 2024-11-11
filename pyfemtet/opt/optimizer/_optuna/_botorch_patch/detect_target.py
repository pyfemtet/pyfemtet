import optuna_integration
from optuna_integration.version import __version__ as optuna_integration_version
from packaging.version import Version


def get_botorch_sampler_module():
    if Version(optuna_integration_version) < Version('4.0.0'):
        return optuna_integration.botorch
    elif Version(optuna_integration_version) < Version('5.0.0'):
        return optuna_integration.botorch.botorch
