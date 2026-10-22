# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

docker pull harbor.sourcefind.cn:5443/hcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12

docker run -itd --name triton_inference_server_ci --privileged --shm-size=16g --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd -v /opt/hyhal:/opt/hyhal:ro --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /public/home/luoxiao/ci:/home/ci harbor.sourcefind.cn:5443/hcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12 bash
docker start triton_inference_server_ci
docker exec -it triton_inference_server_ci bash
