"""
Tests for the command-line interface.
"""

import pytest
from unittest.mock import patch
from leanagent.cli import parse_args, main


def test_parse_args_no_command():
    """Test parsing arguments with no command."""
    args = parse_args([])
    assert args.command is None
    assert args.verbose == 0
    assert args.config is None


def test_parse_args_config_show():
    """Test parsing arguments for config show command."""
    args = parse_args(["config", "--show"])
    assert args.command == "config"
    assert args.show is True
    assert args.set is None


def test_parse_args_config_set():
    """Test parsing arguments for config set command."""
    args = parse_args(["config", "--set", "data", "root_dir", "/new/path"])
    assert args.command == "config"
    assert args.show is False
    assert args.set == [["data", "root_dir", "/new/path"]]


def test_parse_args_run():
    """Test parsing arguments for run command."""
    args = parse_args(["run", "--component", "retrieval"])
    assert args.command == "run"
    assert args.component == "retrieval"


@patch('leanagent.cli.parse_args')
def test_main_config_show(mock_parse_args, capsys):
    """Test main function with config show command."""
    # Create a mock args object
    mock_args = mock_parse_args.return_value
    mock_args.command = "config"
    mock_args.show = True
    mock_args.set = None
    mock_args.config = None
    mock_args.verbose = 0
    
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Current configuration:" in captured.out


@patch('leanagent.cli.parse_args')
def test_main_config_set(mock_parse_args, capsys):
    """Test main function with config set command."""
    # Create a mock args object
    mock_args = mock_parse_args.return_value
    mock_args.command = "config"
    mock_args.show = False
    mock_args.set = [["data", "root_dir", "/test/path"]]
    mock_args.config = None
    mock_args.verbose = 1
    
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Updated configuration:" in captured.out


@patch('leanagent.cli.parse_args')
def test_main_run(mock_parse_args, capsys):
    """Test main function with run command."""
    # Create a mock args object
    mock_args = mock_parse_args.return_value
    mock_args.command = "run"
    mock_args.component = "prover"
    mock_args.config = None
    
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Running component: prover" in captured.out


@patch('leanagent.cli.parse_args')
def test_main_no_command(mock_parse_args, capsys):
    """Test main function with no command."""
    # Create a mock args that returns None for command
    mock_args = mock_parse_args.return_value
    mock_args.command = None
    mock_args.config = None
    
    # Mock the help call to avoid SystemExit
    with patch('leanagent.cli.parse_args', side_effect=[mock_args, None]):
        exit_code = main([])
        assert exit_code == 0


@patch('leanagent.cli.main')
@patch('leanagent.cli.sys.exit')
def test_run_main(mock_exit, mock_main):
    """Test the run_main function."""
    # Setup mock
    mock_main.return_value = 42
    
    # Call the function
    from leanagent.cli import run_main
    run_main()
    
    # Verify the call was made with the correct argument
    mock_exit.assert_called_once_with(42) 