#!/usr/bin/env bash
# Run the test suite under Triton's CPU interpreter inside a Linux container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q -t fusedkernels-cpu -f docker/Dockerfile docker >/dev/null
docker run --rm -v "$PWD":/work -e TRITON_INTERPRET=1 -e PYTHONPATH=/work/src fusedkernels-cpu python -m pytest -q "$@"
