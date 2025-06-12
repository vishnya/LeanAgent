#!/bin/bash
# Run pytest for the LeanAgent package

# Exit on error
set -e

# Make sure we're in the project root
cd "$(dirname "$0")"

# Run pytest with coverage
python -m pytest tests/ -v --cov=leanagent

# Show coverage report
python -m pytest --cov-report term-missing --cov=leanagent tests/ 