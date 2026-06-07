"""Scaffolding sanity check so the test suite collects and CI has a green gate."""

from app import __version__


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"
