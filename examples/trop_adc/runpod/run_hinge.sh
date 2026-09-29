#!/usr/bin/env bash
# Fragment-only hinge remodel. Do not load the full IgG into PyRosetta.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/trastuzumab_project}"
PDB="${PDB:-$WORKDIR/trastuzumab_boltz_model_0.pdb}"
FRAG="${FRAG:-$WORKDIR/B217_238_local.pdb}"
OUT_DIR="${OUT_DIR:-$WORKDIR/B220_235_rosetta}"
N="${N:-5}"
PROTOCOL="${PROTOCOL:-min}"
PY="${PY:-python}"
LOG="${LOG:-$OUT_DIR/hinge_fragment.log}"

mkdir -p "$OUT_DIR"
cd "$WORKDIR"

echo "python: $($PY -c 'import sys; print(sys.executable)')"
echo "full pdb: $PDB"
echo "fragment: $FRAG"
echo "n=$N protocol=$PROTOCOL"

if ! $PY -c "import pyrosetta" 2>/dev/null; then
  echo "PyRosetta missing"
  exit 1
fi

INPUT="$PDB"
if [[ -f "$FRAG" ]]; then
  INPUT="$FRAG"
  echo "using existing fragment $FRAG"
fi

nohup $PY -u hinge_remodel.py --pdb "$INPUT" --out_dir "$OUT_DIR" --n "$N" --protocol "$PROTOCOL" \
  > "$LOG" 2>&1 &
echo "PID=$!"
echo "tail -f $LOG"
echo "When done, download ONLY: $OUT_DIR/B220_235_frag_best.pdb"
echo "Then stitch locally. Do not feed the fragment to adcsim."
