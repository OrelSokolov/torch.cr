#!/usr/bin/env bash
# Locates a usable libtorch distribution and prints:
#   line 1: torch package dir (contains include/ and lib/)
# Exits non-zero if nothing found.
#
# Search order:
#   1. $LIBTORCH (torch package dir or a libtorch zip layout)
#   2. python3/python on PATH with torch importable
#   3. $TORCH_VENV/bin/python
#   4. well-known venvs on this machine
set -euo pipefail

try_python() {
  local py="$1"
  [ -x "$py" ] || return 1
  "$py" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null || return 1
}

if [ -n "${LIBTORCH:-}" ]; then
  # torch package layout: .../torch/include, .../torch/lib
  if [ -d "$LIBTORCH/include" ] && [ -d "$LIBTORCH/lib" ]; then
    echo "$LIBTORCH"; exit 0
  fi
  # libtorch zip layout: <dir>/libtorch/include...
  if [ -d "$LIBTORCH/libtorch/include" ]; then
    echo "$LIBTORCH/libtorch"; exit 0
  fi
  echo "LIBTORCH=$LIBTORCH does not look like a torch install" >&2
  exit 1
fi

for py in python3 python; do
  if d=$(try_python "$(command -v "$py" || true)"); then
    echo "$d"; exit 0
  fi
done

if [ -n "${TORCH_VENV:-}" ]; then
  if d=$(try_python "$TORCH_VENV/bin/python"); then
    echo "$d"; exit 0
  fi
fi

for venv in "$HOME"/dataset-vox-cpm2/.venv "$HOME"/.venv; do
  if d=$(try_python "$venv/bin/python"); then
    echo "$d"; exit 0
  fi
done

echo "libtorch not found; set LIBTORCH or TORCH_VENV" >&2
exit 1
