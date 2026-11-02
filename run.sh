#!/usr/bin/env bash
# Start a sweep detached from the terminal, in its own memory-limited systemd unit.
# If memory runs out, only this calculation is stopped (not your terminal), and
# running the same start command again resumes where it stopped.
#
#   ./run.sh liu cells/liu_cubic.json runs/liu/k.csv      # start or resume
#   ./run.sh liu status                                   # progress
#   ./run.sh liu stop                                     # stop (resumable)
#
# Results and log go to runs/<name>/ under the current directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
name="$1"; unit="brillouin-$name"
venv="${VENV:-$HOME/brillouin-sweep/.venv}"
out="$PWD/runs/$name"
case "${2:-}" in
  status) systemctl --user is-active "$unit" || true; tail -n 3 "$out/log.txt"; exit ;;
  stop)   systemctl --user stop "$unit"; exit ;;
esac
cell="$(realpath "$2")"; kpts="$(realpath "$3")"; shift 3
mkdir -p "$out"
systemd-run --user --unit="$unit" --collect \
  -p MemoryMax="${MEMMAX:-1500M}" -p MemorySwapMax="${SWAPMAX:-512M}" -p Nice=10 \
  -p StandardOutput="append:$out/log.txt" -p StandardError="append:$out/log.txt" \
  --setenv=MKL_NUM_THREADS="${THREADS:-2}" --setenv=OMP_NUM_THREADS=1 \
  --working-directory="$out" \
  "$venv/bin/python" "$here/voxel_fem.py" "$cell" "$kpts" -o "$out/results.csv" --nev 12 --jobs 1 "$@"
echo "started $unit; progress: ./run.sh $name status   (log: $out/log.txt)"
