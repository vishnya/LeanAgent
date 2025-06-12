#!/bin/bash
# LeanAgent Runner Script
#
# This script runs the LeanAgent CLI
#
# Usage: bash run_leanagent.sh [command] [args]
#
# Commands:
#   config - View or modify configuration
#   run - Run a component or the full system
#
# Examples:
#   ./run_leanagent.sh config --show
#   ./run_leanagent.sh run --component retrieval
#   ./run_leanagent.sh run --component all

# Set environment variables if needed
# export LEANAGENT_DATA__ROOT_DIR="/path/to/data"
# export LEANAGENT_PROVER__MODEL="gpt-4"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the CLI module with any arguments passed to the script
python -m leanagent.cli "$@"
