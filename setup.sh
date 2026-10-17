#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or later required"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
USE_GPU=false .venv/bin/python -m pytest -q
printf '%s\n' 'Setup complete. Run: .venv/bin/python -m src.demo'
