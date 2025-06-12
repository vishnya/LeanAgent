"""
Test that imports work correctly.

This test verifies that all package components can be imported without errors.
"""


def test_import_config():
    """Test importing the config module."""
    from leanagent import config
    assert hasattr(config, "Config")
    assert hasattr(config, "get_config")


def test_import_cli():
    """Test importing the CLI module."""
    from leanagent import cli
    assert hasattr(cli, "main")
    assert hasattr(cli, "parse_args") 