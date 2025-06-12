"""
Standalone test for environment variables in configuration.
"""

import os
import pytest
from leanagent.config import Config


def test_env_vars():
    """Test that environment variables are correctly loaded."""
    # Set environment variables
    os.environ["LEANAGENT_DATA__ROOT_DIR"] = "/env/path"
    os.environ["LEANAGENT_RETRIEVAL__K"] = "20"
    os.environ["LEANAGENT_PROVER__TEMPERATURE"] = "0.5"
    
    # Create config instance
    config = Config()
    
    # Check values
    assert config.get("data", "root_dir") == "/env/path"
    assert config.get("retrieval", "k") == 20
    assert config.get("prover", "temperature") == 0.5
    
    # Clean up
    del os.environ["LEANAGENT_DATA__ROOT_DIR"]
    del os.environ["LEANAGENT_RETRIEVAL__K"]
    del os.environ["LEANAGENT_PROVER__TEMPERATURE"] 