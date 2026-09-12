#!/usr/bin/env bash
# Run Timeloop on every generated ws_<shape>_a<array> config, inside the
# accelergy-timeloop Docker image, using the vendored example-design harness.
#
#   Prereq: Docker Desktop running.
#   Image:  timeloopaccelergy/accelergy-timeloop-infrastructure:latest
#           (arm64 build needs LD_LIBRARY_PATH=/usr/local/lib -- set below)
#
# Usage:  validate/run_timeloop.sh  [name-substring-filter]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/bench"
IMAGE="${TIMELOOP_IMAGE:-timeloopaccelergy/accelergy-timeloop-infrastructure:latest}"
FILTER="${1:-}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

ARCHES=$(cd "$BENCH/example_designs" && ls -d ws_*_a* os_*_a* wsw_* 2>/dev/null \
  | { [ -n "$FILTER" ] && grep "$FILTER" || cat; } || true)
if [ -z "$ARCHES" ]; then
  echo "no generated configs. Run: python3 validate/gen_timeloop_configs.py" >&2
  exit 1
fi

for arch in $ARCHES; do
  echo ">>> $arch"
  docker run --rm -v "$BENCH":/work -w /work \
    --entrypoint bash "$IMAGE" -lc "
      export LD_LIBRARY_PATH=/usr/local/lib:\${LD_LIBRARY_PATH:-}
      python3 run_example_designs.py --architecture '$arch' --n_jobs 1
    " 2>&1 | grep -E "Cycles:|Utilization|no valid mapping|error|Error" || true
done

echo "done. stats under $BENCH/example_designs/{ws,os}_*/outputs/default_problem/"
