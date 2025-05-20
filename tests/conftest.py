import pytest
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize


@pytest.fixture(scope='function')
def com_function():
    CoInitialize()
    yield
    CoUninitialize()


def pytest_collection_modifyitems(items, config):
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")
