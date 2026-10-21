#!/bin/bash
# ============================================================
# run_pytorch_backend.sh - PyTorch backend end-to-end test
#   Starts tritonserver (HTTP 8020 / gRPC 8021) with
#   model_repo_pytorch, then calls the pytorch_add (TorchScript)
#   model over HTTP REST.
# usage: bash run_pytorch_backend.sh <WORK_BASE>
# ============================================================
export WORK_BASE=$1
set -e

cd ${WORK_BASE}/triton-inference-server-test || exit 1

P=${WORK_BASE}/opt/triton-server
TORCH=/usr/local/python3.12/lib/python3.12/site-packages/torch
HTTP_PORT=8020
GRPC_PORT=8021
LOG=/tmp/triton_pytorch.log
MODEL_DIR=./model_repo_pytorch/pytorch_add/1

echo "[PT] check pytorch backend files"
if [ ! -f ${P}/backends/pytorch/libtriton_pytorch.so ]; then
    echo "[ERROR] missing ${P}/backends/pytorch/libtriton_pytorch.so (build/install pytorch backend first)"
    exit 1
fi

echo "[PT] ensure TorchScript model"
if [ ! -f ${MODEL_DIR}/model.pt ]; then
    mkdir -p ${MODEL_DIR}
    python3 python/make_pytorch_model.py ${MODEL_DIR}/model.pt
fi

pkill -f tritonserver 2>/dev/null || true
sleep 2

export LD_LIBRARY_PATH=${P}/lib:${TORCH}/lib:/opt/dtk/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH}

echo "[PT] starting tritonserver (HTTP ${HTTP_PORT}) ..."
nohup ${P}/bin/tritonserver \
      --model-repository=./model_repo_pytorch \
      --backend-directory=${P}/backends \
      --http-port=${HTTP_PORT} \
      --grpc-port=${GRPC_PORT} \
      --log-verbose=1 \
      --exit-timeout-secs=10 > ${LOG} 2>&1 &

READY=0
for i in $(seq 1 120); do
    if curl -s -m 2 http://127.0.0.1:${HTTP_PORT}/v2/health/ready >/dev/null 2>&1; then
        echo "[OK] server ready (${i}s)"
        READY=1
        break
    fi
    if ! pgrep -f tritonserver >/dev/null 2>&1; then
        echo "[ERROR] tritonserver exited"
        tail -60 ${LOG}
        exit 1
    fi
    sleep 1
done

if [ "$READY" != "1" ]; then
    echo "[ERROR] HTTP not ready in 120s"
    tail -60 ${LOG}
    pkill -f tritonserver 2>/dev/null || true
    exit 1
fi

echo "----- PyTorch backend end-to-end test -----"
python3 python/test_pytorch_backend.py 127.0.0.1:${HTTP_PORT}

echo "[OK] run_pytorch_backend.sh passed"
pkill -f tritonserver 2>/dev/null || true
