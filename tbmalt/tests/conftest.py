# content of conftest.py
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cpu", help="specify test device (cpu/cuda)"
    )


@pytest.fixture
def device(request) -> torch.device:
    """Defines the device on which each test should be run.

    Returns:
        device: The device on which the test will be run.

    """
    return torch.device(request.config.getoption("--device"))

