#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -eo pipefail

# ---------- 配置 ----------
MODE=${MODE:-auto}
case "${1:-}" in
    --force) MODE=force; shift ;;
    --test-only) MODE=test; shift ;;
esac
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "用法: $0 [--force|--test-only] <宿主机挂载点> <容器内工作目录>" >&2
    exit 1
fi
HOST_CI=$1
WORK_BASE=${2%/}
CONTAINER=${CONTAINER:-triton_inference_server_ci}
IMAGE=${IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12}
TEST_REPO=https://github.com/HYGON-AI/triton-inference-server-test.git
TEST_DIR="$HOST_CI/triton-inference-server-test"
case "${3:-}" in
    --force) MODE=force ;;
    --test-only) MODE=test ;;
    '') ;;
    *) echo "未知模式参数: $3" >&2; exit 1 ;;
esac
case "$MODE" in
    auto|force|test) ;;
    *) echo "未知构建模式: $MODE" >&2; exit 1 ;;
esac
if [[ "$WORK_BASE" != /* || "$WORK_BASE" == / || -z "$WORK_BASE" ]]; then
    echo "容器内工作目录必须是非根目录的绝对路径" >&2
    exit 1
fi

echo "=========================================="
echo " 宿主机一键编排 (模式: $MODE)"
echo "=========================================="

# ---------- [1/5] 获取 test repo ----------
echo "[1/5] test repo -> $TEST_DIR"
if [ "${GITHUB_ACTIONS:-false}" = true ]; then
    # Actions 已 checkout 精确提交（PR 通常是 detached HEAD），不得再次拉取 main。
    if [ "$(realpath -- "$TEST_DIR")" != "$(realpath -- "$GITHUB_WORKSPACE")" ]; then
        echo "CI 必须直接使用 GITHUB_WORKSPACE 中的被测源码" >&2
        exit 1
    fi
    cd "$TEST_DIR"
    echo "[repo] 使用 Actions checkout，不执行 git pull"
elif [ -d "$TEST_DIR/.git" ]; then
    git -C "$TEST_DIR" pull --ff-only
    cd "$TEST_DIR"
else
    echo "  未找到 test repo, 尝试 clone..."
    mkdir -p "$HOST_CI"
    cd "$HOST_CI"
    if git clone "$TEST_REPO" triton-inference-server-test; then
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
if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "创建容器..."
    docker pull "$IMAGE"
    docker run -itd --name "$CONTAINER" --privileged --shm-size=16g \
        --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
        -v /opt/hyhal:/opt/hyhal:ro --group-add video --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined -v "$HOST_CI":"$WORK_BASE" "$IMAGE" bash
fi
docker start "$CONTAINER" >/dev/null

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
CORE_LIB=$WORK_BASE/opt/triton-server/lib/libtritonserver.so
BUILD_MARKER=$WORK_BASE/opt/triton-server/.ci-test-commit
NEED_BUILD=1
if [ "$MODE" = "force" ]; then
    docker exec "$CONTAINER" rm -rf "$WORK_BASE/opt/triton-server"
elif [ "$MODE" = "test" ]; then
    if ! docker exec "$CONTAINER" test -f "$CORE_LIB"; then
        echo "[ERROR] 编译产物不存在，请先运行 auto/force 模式" >&2
        exit 1
    fi
    NEED_BUILD=0
elif [ "$MODE" = "auto" ] && docker exec "$CONTAINER" test -f "$CORE_LIB" 2>/dev/null; then
    if [ "${GITHUB_ACTIONS:-false}" != true ]; then
        NEED_BUILD=0
    elif [ "$(docker exec "$CONTAINER" cat "$BUILD_MARKER" 2>/dev/null || true)" = "${GITHUB_SHA:?}" ]; then
        NEED_BUILD=0
    fi
fi

if [ "$NEED_BUILD" = "1" ]; then
    echo "  运行 build.sh (完整编译 core/backends, 耗时较长)..."
    UPDATE_DEPS=0
    [ "${GITHUB_ACTIONS:-false}" != true ] || UPDATE_DEPS=1
    docker exec -e CI_UPDATE_DEPS="$UPDATE_DEPS" "$CONTAINER" bash -e "$WORK_BASE/triton-inference-server-test/build.sh" "$WORK_BASE" HYGON-AI
    if [ "${GITHUB_ACTIONS:-false}" = true ]; then
        docker exec "$CONTAINER" bash -c 'printf "%s\n" "$1" > "$2"' _ "$GITHUB_SHA" "$BUILD_MARKER"
    fi
else
    echo "  跳过 build.sh (--test-only 或已构建)"
fi

# ---------- [4/5] 双通道及客户端测试 ----------
echo "[4/5] run_test.sh (HTTP/gRPC + C++/Python 端到端测试)"
docker exec "$CONTAINER" bash -e "$WORK_BASE/triton-inference-server-test/run_test.sh" "$WORK_BASE"

# ---------- [5/5] 运行确认 ----------
echo "[5/5] 所有测试命令已成功返回；测试服务已由各测试脚本清理"
echo ""
echo "=========================================="
echo " 完成! 构建及端到端测试通过"
echo "=========================================="
