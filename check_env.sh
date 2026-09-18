#!/usr/bin/env bash
# Run the 3dgs environment smoke test inside the 3dgs_xy container.
# Usage (from host):
#   ./check_env.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker exec 3dgs_xy bash -c "
  source /home/miniconda3/etc/profile.d/conda.sh
  conda activate 3dgs
  export LD_LIBRARY_PATH=/home/miniconda3/envs/3dgs/lib/python3.10/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}
  python /home/3dgs-work/check_env.py
"
