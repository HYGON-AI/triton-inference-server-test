#!/bin/bash
# ============================================================
# run_server.sh - 启动 Triton HTTP(8000)+gRPC(8001) 并端到端测试
# 修复: 服务未就绪时立即报错并退出, 不再静默闯关导致 Connection refused
# ============================================================
export WORK_BASE=$1
set -e

cd ${WORK_BASE}/triton-inference-server-test

echo "[TEST] 启动 HTTP (8000) + gRPC (8001) 服务 ..."
pkill -f tritonserver 2>/dev/null || true
sleep 2
# 确认旧进程已退出、端口已释放 (最多等 10s)
for i in $(seq 1 10); do
    pgrep -f tritonserver >/dev/null 2>&1 || break
    sleep 1
done
export LD_LIBRARY_PATH=${WORK_BASE}/opt/triton-server/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/cuda/cuda-12/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
nohup ${WORK_BASE}/opt/triton-server/bin/tritonserver \
      --model-repository=./model_repo \
      --backend-directory=${WORK_BASE}/opt/triton-server/backends \
      --http-port=8000 \
      --grpc-port=8001 \
      --log-verbose=0 \
      --exit-timeout-secs=10 > /tmp/triton_http.log 2>&1 &

echo "等待 HTTP 服务就绪 ..."
READY=0
for i in $(seq 1 90); do
    if curl -s -m 2 http://127.0.0.1:8000/v2/health/ready >/dev/null 2>&1; then
        echo "[OK] HTTP 服务就绪 (约 ${i}s)"
        READY=1
        break
    fi
    # 进程已退出: 启动失败, 立即报错
    if ! pgrep -f tritonserver >/dev/null 2>&1; then
        echo "[ERROR] tritonserver 进程已退出, 启动失败"
        echo "----- /tmp/triton_http.log (尾部) -----"
        tail -40 /tmp/triton_http.log
        exit 1
    fi
    sleep 1
done

if [ "$READY" != "1" ]; then
    echo "[ERROR] 90s 内 HTTP 服务未就绪"
    echo "----- /tmp/triton_http.log (尾部) -----"
    tail -40 /tmp/triton_http.log
    exit 1
fi

cd ${WORK_BASE}/triton-inference-server-test
echo "----- HTTP 端到端测试 -----"
python3 python/test_yolo_http.py http://127.0.0.1:8000 samples/bus_input.bin
echo "----- gRPC 端到端测试 -----"
python3 python/test_yolo_grpc.py 127.0.0.1:8001 samples/bus_input.bin
echo "[OK] run_server.sh 全部测试通过"

ps -ef|grep tritonserver |awk '{print $2}' |xargs kill -9
