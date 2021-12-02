import pytest 

def pytest_addoption(parser):
    parser.addoption("--external_delegate", action="store", default="none")

@pytest.fixture(scope='session')
def delegate_lib(request):
    delegate_path = request.config.option.external_delegate
    if delegate_path is None:
        pytest.skip()
    return delegate_path 