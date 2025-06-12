"""
Tests for the configuration management functionality.
"""

import os
import pytest
from pathlib import Path
import copy

from leanagent.config import Config, get_config


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton between tests."""
    # Save the original _config
    from leanagent.config import _config
    original_config = _config
    
    # Reset for the test
    import leanagent.config
    leanagent.config._config = None
    
    # Run the test
    yield
    
    # Restore the original _config
    leanagent.config._config = original_config


def test_default_config(monkeypatch):
    """Test that the default configuration is loaded correctly."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_RETRIEVAL__K", raising=False)
    monkeypatch.delenv("LEANAGENT_PROVER__TEMPERATURE", raising=False)
    
    # Create a clean instance of Config with the DEFAULT_CONFIG
    config = Config()
    default_config = copy.deepcopy(Config.DEFAULT_CONFIG)
    
    # Test against the DEFAULT_CONFIG directly
    assert config.get("data", "root_dir") == default_config["data"]["root_dir"]
    assert config.get("retrieval", "k") == default_config["retrieval"]["k"]
    assert config.get("prover", "temperature") == default_config["prover"]["temperature"]


def test_config_from_file(temp_config_file, monkeypatch):
    """Test loading configuration from a file."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_RETRIEVAL__K", raising=False)
    
    config = Config(temp_config_file)
    assert config.get("data", "root_dir") == "./test_data"
    assert config.get("retrieval", "k") == 5
    assert config.get("retrieval", "similarity_threshold") == 0.8
    # Default values should still be available for non-overridden settings
    assert config.get("prover", "temperature") == 0.7


def test_config_with_invalid_file(monkeypatch):
    """Test loading configuration with an invalid file path."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_RETRIEVAL__K", raising=False)
    
    config = Config("nonexistent_file.yaml")
    # Should fall back to defaults
    default_config = copy.deepcopy(Config.DEFAULT_CONFIG)
    assert config.get("data", "root_dir") == default_config["data"]["root_dir"]
    assert config.get("retrieval", "k") == default_config["retrieval"]["k"]


def test_set_and_get(monkeypatch):
    """Test setting and getting configuration values."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    
    config = Config()
    
    # Test setting a value
    config.set("data", "new_option", "new_value")
    assert config.get("data", "new_option") == "new_value"
    
    # Test default value for non-existent option
    assert config.get("data", "nonexistent", "default") == "default"
    
    # Test setting a value in a new section
    config.set("new_section", "option", "value")
    assert config.get("new_section", "option") == "value"


def test_update_from_dict(monkeypatch):
    """Test updating configuration from a dictionary."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_RETRIEVAL__K", raising=False)
    
    # Start with a fresh config
    config = Config()
    default_config = copy.deepcopy(Config.DEFAULT_CONFIG)
    
    # Verify initial state
    assert config.get("data", "root_dir") == default_config["data"]["root_dir"]
    assert config.get("retrieval", "k") == default_config["retrieval"]["k"]
    
    update_dict = {
        "data": {
            "root_dir": "/updated/path",
            "new_option": "new_value"
        },
        "new_section": {
            "option": "value"
        }
    }
    
    config.update_from_dict(update_dict)
    
    assert config.get("data", "root_dir") == "/updated/path"
    assert config.get("data", "new_option") == "new_value"
    assert config.get("new_section", "option") == "value"
    # Non-updated values should remain
    assert config.get("retrieval", "k") == default_config["retrieval"]["k"]


def test_get_config_singleton(monkeypatch):
    """Test the get_config singleton function."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    
    config1 = get_config()
    config2 = get_config()
    
    # They should be the same instance
    assert config1 is config2
    
    # Modify through one instance
    config1.set("test", "option", "value")
    
    # Check through the other instance
    assert config2.get("test", "option") == "value"


def test_as_dict(monkeypatch):
    """Test getting the full configuration as a dictionary."""
    # Clear any environment variables that might affect the test
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    
    config = Config()
    default_config = copy.deepcopy(Config.DEFAULT_CONFIG)
    config_dict = config.as_dict()
    
    assert isinstance(config_dict, dict)
    assert "data" in config_dict
    assert "root_dir" in config_dict["data"]
    
    # Modify the copy
    original_value = config.get("data", "root_dir")
    config_dict["data"]["root_dir"] = "/changed/path"
    
    # Original should be unchanged
    assert config.get("data", "root_dir") == original_value


def test_convert_value():
    """Test converting string values to appropriate types."""
    config = Config()
    
    # Test boolean conversion
    assert config._convert_value("true") is True
    assert config._convert_value("True") is True
    assert config._convert_value("yes") is True
    assert config._convert_value("1") is True
    
    assert config._convert_value("false") is False
    assert config._convert_value("False") is False
    assert config._convert_value("no") is False
    assert config._convert_value("0") is False
    
    # Test number conversion
    assert config._convert_value("42") == 42
    assert config._convert_value("3.14") == 3.14
    
    # Test string values
    assert config._convert_value("hello") == "hello"


def test_update_nested_dict():
    """Test updating nested dictionaries."""
    config = Config()
    
    target = {
        "a": {
            "b": 1,
            "c": 2
        },
        "d": 3
    }
    
    source = {
        "a": {
            "b": 10,
            "e": 20
        },
        "f": 30
    }
    
    # Create a deep copy to ensure we don't affect other tests
    target_copy = copy.deepcopy(target)
    
    config._update_nested_dict(target_copy, source)
    
    assert target_copy == {
        "a": {
            "b": 10,
            "c": 2,
            "e": 20
        },
        "d": 3,
        "f": 30
    }


def test_environment_variables(monkeypatch):
    """Test loading configuration from environment variables."""
    # First clear any existing env vars that might interfere
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_RETRIEVAL__K", raising=False)
    monkeypatch.delenv("LEANAGENT_PROVER__TEMPERATURE", raising=False)
    
    # Explicitly reset the environment variable cache
    import os
    
    # Set environment variables
    os.environ["LEANAGENT_DATA__ROOT_DIR"] = "/env/path"
    os.environ["LEANAGENT_RETRIEVAL__K"] = "20"
    os.environ["LEANAGENT_PROVER__TEMPERATURE"] = "0.5"
    
    # Create a fresh config after setting environment variables
    config = Config()
    
    # Check values were loaded from environment
    assert config.get("data", "root_dir") == "/env/path"
    assert config.get("retrieval", "k") == 20
    assert config.get("prover", "temperature") == 0.5
    
    # Clean up
    del os.environ["LEANAGENT_DATA__ROOT_DIR"]
    del os.environ["LEANAGENT_RETRIEVAL__K"]
    del os.environ["LEANAGENT_PROVER__TEMPERATURE"]


def test_environment_variable_new_section(monkeypatch):
    """Test loading environment variables into a new section."""
    # Clear any existing env vars that might interfere
    monkeypatch.delenv("LEANAGENT_DATA__ROOT_DIR", raising=False)
    monkeypatch.delenv("LEANAGENT_NEW_SECTION__OPTION", raising=False)
    
    # Create a config instance
    config = Config()
    
    # Set environment variable for new section that doesn't exist in config
    os.environ["LEANAGENT_NEW_SECTION__OPTION"] = "test_value"
    
    # Manually call _load_from_env to test the condition
    config._load_from_env()
    
    # Verify the section was NOT created (since it doesn't exist in default config)
    assert "new_section" not in config.config
    
    # Now create the section but without initializing it
    config.config["new_section"] = None  # This is not a dict, to trigger the condition
    
    # Call _load_from_env again
    config._load_from_env()
    
    # Check if the section was properly initialized as a dict
    assert isinstance(config.config["new_section"], dict)
    assert config.get("new_section", "option") == "test_value"
    
    # Clean up
    del os.environ["LEANAGENT_NEW_SECTION__OPTION"] 