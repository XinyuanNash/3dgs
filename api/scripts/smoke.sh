#!/usr/bin/env bash
# 3DGS Pipeline Service — 烟雾测试脚本
#
# 用法:
#   1. 启动服务:
#      cd /data4/huxinyuan/3dgs/api
#      /usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 &
#
#   2. 准备 ~20 张测试图片到 /tmp/smoke_input/
#      mkdir -p /tmp/smoke_input
#      cp /data4/huxinyuan/3dgs/3dgs-work/datasets/campus/images/IMG_*.JPG /tmp/smoke_input/ | head -20
#
#   3. 运行此脚本:
#      bash /data4/huxinyuan/3dgs/api/scripts/smoke.sh
#
# 它会:
#   - 健康检查
#   - 上传 → 创建 job
#   - 轮询直到 SUCCEEDED 或 60 分钟超时
#   - 下载最终 PLY
set -euo pipefail

API="${API:-http://localhost:8000/api/v1}"
INPUT_DIR="${INPUT_DIR:-/tmp/smoke_input}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-60}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[smoke] INPUT_DIR does not exist: $INPUT_DIR"
    echo "[smoke] create it and copy some images first"
    exit 1
fi

echo "[smoke] === 1. health check ==="
curl -fsS "$API/health" && echo

echo "[smoke] === 2. submit job ==="
# 使用 curl 的 -F 多文件上传
JOB_JSON=$(mktemp)
curl -fsS -X POST "$API/jobs" \
    -F "name=smoke_$(date +%H%M%S)" \
    -F "iterations=30000" \
    $(for f in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.JPG "$INPUT_DIR"/*.png; do
        [[ -f "$f" ]] && echo -n " -F files=@$f"
      done) \
    -o "$JOB_JSON"
cat "$JOB_JSON"; echo

JOB_ID=$(python3 -c "import json; print(json.load(open('$JOB_JSON'))['id'])")
echo "[smoke] job_id=$JOB_ID"

echo "[smoke] === 3. poll until SUCCEEDED (max ${MAX_WAIT_MIN} min) ==="
DEADLINE=$((SECONDS + MAX_WAIT_MIN * 60))
while (( SECONDS < DEADLINE )); do
    STATUS=$(curl -fsS "$API/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "[smoke] $(date +%H:%M:%S) status=$STATUS"
    if [[ "$STATUS" == "succeeded" ]]; then
        echo "[smoke] SUCCEEDED!"
        break
    fi
    if [[ "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then
        echo "[smoke] job ended with status=$STATUS"
        curl -fsS "$API/jobs/$JOB_ID" | python3 -m json.tool
        exit 1
    fi
    sleep "$POLL_INTERVAL"
done

if (( SECONDS >= DEADLINE )); then
    echo "[smoke] TIMEOUT after ${MAX_WAIT_MIN} min"
    exit 1
fi

echo "[smoke] === 4. download PLY ==="
PLY_PATH="/tmp/smoke_${JOB_ID}.ply"
curl -fsS "$API/jobs/$JOB_ID/ply" -o "$PLY_PATH"
ls -lh "$PLY_PATH"

echo "[smoke] === 5. validate PLY ==="
python3 -c "
from plyfile import PlyData
p = PlyData.read('$PLY_PATH')
v = p['vertex']
print(f'vertex count: {len(v)}')
print(f'properties: {[pr.name for pr in v.properties[:6]]}...')
"

echo "[smoke] === 6. final status ==="
curl -fsS "$API/jobs/$JOB_ID" | python3 -m json.tool

echo "[smoke] DONE"