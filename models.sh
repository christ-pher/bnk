#!/usr/bin/env bash
# Manage bnk's models: ./models.sh list | download NAME [--dir DIR] | add REPO [--include PATTERN]   (tools/models.py)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
fi
exec "$HERE/.venv/bin/python" "$HERE/tools/models.py" "$@"
