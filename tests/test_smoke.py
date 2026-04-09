"""Smoke test to verify the test harness runs."""

import grace


def test_package_version_is_defined() -> None:
    assert grace.__version__ == "0.1.0"
