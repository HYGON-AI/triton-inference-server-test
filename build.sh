#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# build.sh - 一键安装依赖并编译
#   1) server      : triton-inference-server-server-das
#                    (编译时自动从 forwithgod 拉取 common/core/backend)
#   2) onnx backend: triton-inference-server-onnxruntime-backend
#   3) migraphx    : triton-inference-server-migraphx-backend
#
# 用法: bash build.sh <WORK_BASE>
#   安装结果沿用原有目录布局，运行脚本无需修改:
#     <WORK_BASE>/opt/triton-server/
#     <WORK_BASE>/opt/triton-server/backends  -> libtriton_onnx.so / libtriton_migraphx.so
# ============================================================
export WORK_BASE=$1
. "$(dirname "$(realpath "$0")")/env.sh"

TRITON_ORG=https://github.com/$2
echo "TRITON_ORG:$TRITON_ORG"
# tracing 依赖 OpenTelemetry（多层 github 下载），默认关闭；需要时 TRITON_ENABLE_TRACING=ON
TRITON_ENABLE_TRACING=${TRITON_ENABLE_TRACING:-OFF}
SRC_ROOT=${WORK_BASE}/triton_inference_server
PREFIX=${WORK_BASE}/opt/triton-server
SERVER_PREFIX=${PREFIX}
BACKEND_PREFIX=${PREFIX}
CACHE_CORE=${SRC_ROOT}/cache_core

set -e

cd "$WORK_BASE"

# ============================================================
# 1) 依赖安装（与旧脚本一致；已安装的会跳过）
# ============================================================
echo "===== [1/2] 依赖安装 ====="

# migraphx
if [ ! -e /opt/dtk/lib/libmigraphx_c.so.3 ]; then
  echo "[deps] install migraphx"
  curl -L https://download.sourcefind.cn:65024/file/4/migraphx/DAS1.8.4/migraphx-5.2.1+das.opt1.dtk2604-cp310-cp310-manylinux_2_28_x86_64.run -o migraphx.run
  bash ./migraphx.run
fi

# onnxruntime (1.19.2)
if ! python3 -c "import onnxruntime,sys; sys.exit(0 if onnxruntime.__version__=='1.19.2' else 1)" >/dev/null 2>&1; then
  echo "[deps] install onnxruntime"
  wget -nc https://download.sourcefind.cn:65024/file/4/onnxruntime/DAS1.8/onnxruntime-1.19.2+das.opt1.dtk2604.torch271-cp312-cp312-manylinux_2_28_x86_64.whl
  pip uninstall numpy -y || true
  pip install "numpy<2"
  pip install ./*.whl
fi

# cmake 3.31.8
if [ ! -x /opt/cmake-3.31.8/bin/cmake ]; then
  echo "[deps] install cmake 3.31.8"
  wget -nc http://42.228.13.241:18000/thirdparty/triton_inference_server/r26.09/cmake-3.31.8-linux-x86_64.tar.gz
  tar -zxvf cmake-3.31.8-linux-x86_64.tar.gz
  mv cmake-3.31.8-linux-x86_64 /opt/cmake-3.31.8
  ln -sf /opt/cmake-3.31.8/bin/* /usr/local/bin/
fi

pip install --upgrade setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3.12 install --upgrade pip
pip install build wheel setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
apt-get update
apt-get install -y rapidjson-dev libprotobuf-dev protobuf-compiler
apt-get install -y build-essential g++ python3-dev libbz2-dev zlib1g-dev libssl-dev wget
apt-get install -y libb64-dev git-lfs

# boost 1.82
if [ ! -e /usr/local/lib/libboost_json.a ]; then
  echo "[deps] install boost 1.82"
  curl -L -o boost_1_82_0.tar.gz http://42.228.13.241:18000/thirdparty/triton_inference_server/r26.09/boost_1_82_0.tar.gz
  tar -xvf boost_1_82_0.tar.gz
  (cd boost-1.82.0 && ./bootstrap.sh --prefix=/usr/local && ./b2 install -j"$(nproc)" && ldconfig)
fi

# ============================================================
# 2) 拉取源码 + 编译 server / 两个 backend
# ============================================================
echo "===== [2/2] 编译 server 和两个 backend ====="

mkdir -p "$SRC_ROOT"
cd "$SRC_ROOT"

clone_repo() {
  local url=$1 dir=$2
  if [ -d "$dir/.git" ]; then
    echo "[repo] reuse $dir"
  else
    echo "[repo] clone $url -> $dir"
    git clone "$url" "$dir"
  fi
}

clone_repo "$TRITON_ORG/triton-inference-server-server-das.git"            triton-inference-server-server-das
clone_repo "$TRITON_ORG/triton-inference-server-onnxruntime-backend.git"    triton-inference-server-onnxruntime-backend
clone_repo "$TRITON_ORG/triton-inference-server-migraphx-backend.git"       triton-inference-server-migraphx-backend
clone_repo "$TRITON_ORG/triton-inference-server-pytorch-backend-das.git"    triton-inference-server-pytorch-backend-das

mkdir -p "$PREFIX"

# ---------- 2.1 server（自动拉取 common/core/backend） ----------
echo "----- build server -----"
cd triton-inference-server-server-das
rm -rf build && mkdir build && cd build

CMAKE_ARGS=(
  ..
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc
  -DUSE_HCU=ON
  -DTRITON_REPO_ORGANIZATION="$TRITON_ORG"
  -DTRITON_ENABLE_GPU=ON
  -DTRITON_MIN_COMPUTE_CAPABILITY=7.0
  -DTRITON_ENABLE_GRPC=ON
  -DTRITON_ENABLE_HTTP=ON
  -DTRITON_ENABLE_TRACING=${TRITON_ENABLE_TRACING}
  -DTRITON_ENABLE_NVTX=ON
  -DTRITON_ENABLE_STATS=ON
  -DTRITON_ENABLE_METRICS=ON
  -DTRITON_ENABLE_METRICS_CPU=ON
  -DTRITON_ENABLE_METRICS_GPU=OFF
  -DTRITON_ENABLE_GCS=OFF
  -DTRITON_ENABLE_S3=OFF
  -DTRITON_ENABLE_AZURE_STORAGE=OFF
  -DTRITON_ENABLE_ENSEMBLE=OFF
  -DCMAKE_INSTALL_PREFIX="$SERVER_PREFIX"
)
# 并清掉旧 stamp/构建目录，让它按本次安装前缀重新 configure/build/install。
if [ -d "$CACHE_CORE/repo-third-party-build" ]; then
  echo "[cache] reuse third-party source/build cache: $CACHE_CORE"
  rm -rf "$CACHE_CORE"/repo-third-party-build/*/src/*-stamp/*-install
  rm -rf "$CACHE_CORE"/repo-third-party-build/*/src/*-stamp/*-build
  rm -rf "$CACHE_CORE"/repo-third-party-build/*/src/*-stamp/*-configure
  rm -rf "$CACHE_CORE"/repo-third-party-build/*/src/*-build
  # 避免复用绑定旧安装前缀的 core 构建缓存
  rm -rf "$CACHE_CORE"/repo-core-build
  CMAKE_ARGS+=(
    -DFETCHCONTENT_BASE_DIR="$CACHE_CORE"
    -DFETCHCONTENT_SOURCE_DIR_REPO-THIRD-PARTY="$CACHE_CORE/repo-third-party-src"
  )
fi
cmake "${CMAKE_ARGS[@]}"
make -j"$(nproc)"
make install
cd ../..

# ---------- 2.2 onnxruntime backend ----------
echo "----- build onnxruntime backend -----"
cd triton-inference-server-onnxruntime-backend
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DDTK_PATH=/opt/dtk \
  -DCMAKE_PREFIX_PATH=/opt/dtk \
  -DTRITON_CORE_PATH="$SERVER_PREFIX" \
  -DTRITON_BACKEND_PATH="$BACKEND_PREFIX/lib" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
make -j"$(nproc)"
make install
cd ../..

# ---------- 2.3 migraphx backend ----------
echo "----- build migraphx backend -----"
cd triton-inference-server-migraphx-backend
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DDTK_PATH=/opt/dtk \
  -DCMAKE_PREFIX_PATH=/opt/dtk \
  -DTRITON_CORE_PATH="$SERVER_PREFIX" \
  -DTRITON_BACKEND_PATH="$BACKEND_PREFIX/lib" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
make -j"$(nproc)"
make install
cd ../..


# ---------- 2.4 pytorch backend ----------
if [ -d "$SRC_ROOT/triton-inference-server-pytorch-backend-das" ]; then
  echo "----- build pytorch backend -----"
  cd triton-inference-server-pytorch-backend-das
  rm -rf build && mkdir build && cd build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
    -DUSE_HCU=ON \
    -DTRITON_ENABLE_GPU=ON \
    -DTRITON_ENABLE_STATS=ON \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DTRITON_CORE_PATH="$PREFIX" \
    -DTRITON_BACKEND_PATH="$PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
  make -j"$(nproc)"
  make install
  cd ../..
else
  echo "[skip] pytorch backend source not found"
fi


# ============================================================
# 3) Python 依赖（tritonserver 绑定 + tritonclient）
# ============================================================
echo "===== 安装 Python 包 ====="
TRITONSERVER_WHL=$(ls "$SERVER_PREFIX"/python/tritonserver*.whl "$CACHE_CORE"/repo-core-build/python/generic/wheel/dist/tritonserver*.whl 2>/dev/null | head -1)
[ -n "$TRITONSERVER_WHL" ] && pip install --force-reinstall "$TRITONSERVER_WHL" || echo "[WARN] tritonserver wheel not found"
[ -f "$SERVER_PREFIX"/python/tritonfrontend*.whl ] && pip install "$SERVER_PREFIX"/python/tritonfrontend*.whl || true
pip install "tritonclient[grpc]" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[OK] build 完成"
echo "     tritonserver : $SERVER_PREFIX/bin/tritonserver"
echo "     onnx backend : $PREFIX/backends/onnx/libtriton_onnx.so"
echo "     migraphx     : $PREFIX/backends/migraphx/libtriton_migraphx.so"
echo "     pytorch      : $PREFIX/backends/pytorch/libtriton_pytorch.so"
