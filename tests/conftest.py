"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from leanagent.config import Config


@pytest.fixture
def temp_config_file():
    """Fixture providing a temporary config file."""
    config_content = """
data:
  root_dir: "./test_data"
  repo_dir: "./test_data/repos"
retrieval:
  k: 5
  similarity_threshold: 0.8
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    os.unlink(config_path) 