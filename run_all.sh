#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -e

# ---------- 配置 ----------
HOST_CI=$1         # 宿主机挂载点 (挂到容器 /home/ci)
WORK_BASE=$2
CONTAINER=triton_inference_server_ci
IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12
TEST_REPO=https://developer.sourcefind.cn/codes/OpenDAS/triton_inference_server_test.git
TEST_DIR="$HOST_CI/triton_inference_server_test"
MODE="auto"
[ "$1" = "--force" ] && MODE="force"
[ "$1" = "--test-only" ] && MODE="test"

echo "=========================================="
echo " 宿主机一键编排 (模式: $MODE)"
echo "=========================================="

# ---------- [1/5] 获取 test repo ----------
echo "[1/5] test repo -> $TEST_DIR"
if [ -d "$TEST_DIR/.git" ]; then
    cd "$TEST_DIR" && git pull 2>&1 | tail -1 || echo "  [提示] git pull 失败 (宿主机无凭据?), 使用现有代码继续"
else
    echo "  未找到 test repo, 尝试 clone..."
    mkdir -p "$HOST_CI"
    cd "$HOST_CI"
    if git clone "$TEST_REPO" triton_inference_server_test 2>/dev/null; then
        echo "  clone 成功"
    else
        echo "  [错误] 宿主机 git clone 失败. 请先配置凭据:"
        echo "    git config --global credential.helper store"
        echo "    echo 'https://<user>:<token>@developer.sourcefind.cn' > ~/.git-credentials"
        echo "    或绑定 SSH key 到 sourcefind"
        exit 1
    fi
fi

# ---------- [2/5] 创建/启动容器 ----------
echo "[2/5] 容器创建/启动 ($CONTAINER)"
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "创建容器..."
    docker pull "$IMAGE"
    docker run -itd --name "$CONTAINER" --privileged --shm-size=16g \
        --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
        -v /opt/hyhal:/opt/hyhal:ro --group-add video --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined -v "$HOST_CI":"$WORK_BASE" "$IMAGE" bash
fi
docker start "$CONTAINER" >/dev/null 2>&1 || true

# ---------- [3/5] 容器内环境准备 + 编译 (build.sh) ----------
echo "[3/5] 容器内环境准备 + build.sh 编译"
docker exec "$CONTAINER" bash -c "
    # git 凭据: 容器已有 /root/.git-credentials 则复用, 不写入密码
    if [ ! -f /root/.git-credentials ]; then
        echo '[提示] 容器无 /root/.git-credentials, 需手动配置后重试:'
        echo '  echo \"https://<user>:<token>@developer.sourcefind.cn\" > /root/.git-credentials'
    fi
    git config --global url.'https://ghfast.top/https://github.com/'.insteadOf 'https://github.com/'
   
"

# 判断是否需要完整 build.sh (core/backends 是否已构建)
CORE_LIB=$WORK_BASE/opt/triton-server/core/lib/libtritonserver.so
NEED_BUILD=1
if [ "$MODE" = "test" ]; then
    NEED_BUILD=0
elif [ "$MODE" = "auto" ] && docker exec "$CONTAINER" test -f "$CORE_LIB" 2>/dev/null; then
    echo "  [检测] core 已构建 ($CORE_LIB), 跳过 build.sh"
    NEED_BUILD=0
fi

if [ "$NEED_BUILD" = "1" ]; then
    echo "  运行 build.sh (完整编译 core/backends, 耗时较长)..."
    docker exec "$CONTAINER" bash $WORK_BASE/triton_inference_server_test/build.sh $WORK_BASE HYGON-AI
else
    echo "  跳过 build.sh (--test-only 或已构建)"
fi

# ---------- [4/5] 编译 server + 双通道测试 ----------
echo "[4/5] build_server.sh test (server 编译 + HTTP/gRPC 端到端测试)"
docker exec "$CONTAINER" bash $WORK_BASE/triton_inference_server_test/run_test.sh $WORK_BASE

# ---------- [5/5] 运行确认 ----------
echo "[5/5] 运行确认"
sleep 2
echo "HTTP 健康检查:"
docker exec "$CONTAINER" curl -s -m 5 http://127.0.0.1:8000/v2/health/ready && echo " (HTTP 200, 服务正常)"
echo ""
echo "=========================================="
echo " 完成! 服务已启动:"
echo "   HTTP : http://42.228.13.157:8000"
echo "   gRPC : 42.228.13.157:8001"
echo "   推理 : python3 python/test_yolo_http.py http://42.228.13.157:8000"
echo "=========================================="
