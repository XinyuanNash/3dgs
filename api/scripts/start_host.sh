#!/usr/bin/env bash
# 从宿主机启动 3DGS Pipeline Service
#
# 设计:
#   - uvicorn 跑在宿主机(这里)
#   - 通过 docker exec 调用 3dgs_xy 容器内的 colmap/train.py
#   - 因为容器内没有 docker,所以 uvicorn 不能跑在容器里
#
# FastAPI 栈是从 3dgs_xy 容器内 docker cp 出来的 (host-site-packages/)
# 因为宿主机无法访问 pypi (pypi.org/清华/阿里云都不通)
#
# 用法:
#   bash /data4/huxinyuan/3dgs/api/scripts/start_host.sh          # 前台
#   bash /data4/huxinyuan/3dgs/api/scripts/start_host.sh --bg     # 后台
set -euo pipefail

API_DIR=/data4/huxinyuan/3dgs/api
PY=/data4/huxinyuan/3dgs/miniconda3/envs/3dgs/bin/python
HOST_SITE=$API_DIR/host-site-packages
PORT="${PORT:-8005}"

export PYTHONPATH="$HOST_SITE${PYTHONPATH:+:$PYTHONPATH}"

# 检查 docker 可用
if ! command -v docker >/dev/null 2>&1; then
    echo "FATAL: docker not in PATH" >&2
    exit 1
fi

# 检查 3dgs_xy 容器运行中
if ! docker ps --format '{{.Names}}' | grep -qx '3dgs_xy'; then
    echo "FATAL: 3dgs_xy 容器未运行" >&2
    echo "  docker start 3dgs_xy" >&2
    exit 1
fi

cd "$API_DIR"

echo "==============================================="
echo " 3DGS Pipeline Service (host)"
echo "==============================================="
echo "Python:    $PY ($($PY --version 2>&1))"
echo "FastAPI:   $(PYTHONPATH=$HOST_SITE $PY -c 'import fastapi; print(fastapi.__version__)' 2>&1)"
echo "Port:      $PORT"
echo "Container: $(docker ps --format '{{.Names}}\t{{.Image}}' | grep '^3dgs_xy')"
echo "==============================================="

if [[ "${1:-}" == "--bg" ]]; then
    nohup $PY -m uvicorn app:app --host 0.0.0.0 --port "$PORT" \
        > "$API_DIR/server.log" 2>&1 &
    echo "PID: $!"
    echo "Log: $API_DIR/server.log"
    sleep 2
    if kill -0 $! 2>/dev/null; then
        echo "OK: service started"
    else
        echo "FAIL: see $API_DIR/server.log" >&2
        tail -20 "$API_DIR/server.log" >&2
        exit 1
    fi
else
    exec $PY -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
fi
