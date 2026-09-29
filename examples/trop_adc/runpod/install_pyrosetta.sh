#!/usr/bin/env bash
# Install academic/non-commercial PyRosetta into THIS Python.
# Official docs: https://www.pyrosetta.org/downloads
#
# Academic / non-profit / government: free under the PyRosetta Non-Commercial License.
# Commercial use needs a paid license from UW CoMotion (license@uw.edu).
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "Python: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "Version: $($PYTHON -c 'import sys; print(sys.version)')"

if $PYTHON -c "import pyrosetta" 2>/dev/null; then
  echo "PyRosetta already importable. Skip."
  $PYTHON -c "import pyrosetta; print(pyrosetta.__file__)"
  exit 0
fi

echo
echo "Installing quarterly PyRosetta release (west mirror)..."
echo "This is a large wheel (~1–2 GB). Do not use plain 'pip install pyrosetta' from PyPI."
echo

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release

echo
echo "Verify:"
$PYTHON - <<'PY'
import sys
import pyrosetta
print("import OK")
print("python:", sys.executable)
pyrosetta.init("-mute all")
print("init OK")
PY
