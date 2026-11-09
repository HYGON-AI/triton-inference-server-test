# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

export CPLUS_INCLUDE_PATH=/opt/dtk/roctracer/include:${CPLUS_INCLUDE_PATH}
export LIBRARY_PATH=/opt/dtk/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/dtk/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${WORK_BASE}/opt/triton-server/lib:/opt/dtk/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH

export CMAKE_PREFIX_PATH=/opt/dtk/hip:/opt/dtk:${CMAKE_PREFIX_PATH}

# Resolve Torch from the active Python, including images using a virtualenv.
if TORCH_LIB=$(python3 -c 'from importlib.metadata import distribution; print(distribution("torch").locate_file("torch/lib"))' 2>/dev/null); then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

