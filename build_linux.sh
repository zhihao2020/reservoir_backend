#!/usr/bin/env bash
# Compile only src/ to src.so (Linux). numpy/scipy/PyYAML remain pip deps.
#
#   sudo apt install build-essential python3-dev ccache
#   pip install -r requirements.txt
#   ./build_linux.sh
#
# Artifact: dist/linux/src.so  and  dist/linux/reservoir-backend-linux-so.zip
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/build_src_so.py "$@"
