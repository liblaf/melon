import pytest
import warp as wp


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verbose_warnings = True
    wp.init()
