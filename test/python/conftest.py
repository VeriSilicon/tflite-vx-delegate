import pytest
import os

def pytest_addoption(parser):
    parser.addoption("--external_delegate", action="store", default="none")
    parser.addoption("--save_test_model", action="store", default="none" )

@pytest.fixture(scope='session')
def delegate_lib(request):
    delegate_path = request.config.option.external_delegate
    if delegate_path is None:
        pytest.skip()
    return delegate_path

@pytest.fixture(scope='session')
def save_model(request):
    save_model_dir= request.config.option.save_test_model
    if save_model_dir is None :
        pytest.skip()
    return save_model_dir