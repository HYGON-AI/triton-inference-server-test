# Triton Inference Server (HCU) 测试套件

在**海光 HCU**平台上运行与验证 Triton Inference Server（core + backend 移植版）的端到端测试套件，覆盖 **onnx** 与 **migraphx** 两个推理后端，包含模型加载、精度校验、批量性能、多请求合并（dynamic batching）等测试。

- 制品仓库：`https://github.com/HYGON-AI/triton-inference-server-test.git`
- 依赖源码：core / backend / onnx_backend / migraphx_backend / server（均由 `build.sh` 拉取并编译）

---

# 上游归属
- 本仓库是面向 HCU 平台的 Triton Inference Server 适配测试套件，并非上游源码的完整镜像。
- Modified by Hygon Information Technology Co., Ltd., 2026.
---

## 一、目录结构

```
triton-inference-server-test/
├── build.sh                     # 一键构建: 装依赖 + 编译 core/backends/server + 装 Python 包
├── run_all.sh                   # 宿主机一键编排: clone → 起容器 → build.sh → run_test.sh
├── run_test.sh                  # 容器内: 跑所有测试 (C++ + Python in-process)
├── run_server.sh                # 启动 HTTP(8000)/gRPC(8001) 服务 + 端到端测试
├── docker_create.sh             # 单独创建容器 (调试用)
├── env.sh                       # 环境变量 (LD_LIBRARY_PATH / CMAKE_PREFIX_PATH 等)
├── README.md
│
├── src/                         # C++ 源码 (可执行产物编译到仓库根目录)
│   ├── test_backend_infer.cc    #   C++ C-API 客户端 (ResNet50)
│   └── test_yolo.cc             #   C++ C-API 客户端 (YOLOv5s)
│
├── python/                      # Python 测试脚本
│   ├── triton_compat.py         #   in-process 加载 backend 的兼容处理 (见 §6)
│   ├── build_yolo_input.py      #   图片 → YOLO FP16 输入张量
│   ├── test_yolo.py             #   in-process 客户端 (YOLO)
│   ├── test_yolo_batch.py       #   YOLO 批量性能测试
│   ├── test_yolo_http.py        #   HTTP 端到端客户端 (YOLO)
│   ├── test_yolo_grpc.py        #   gRPC 端到端客户端 (YOLO)
│   ├── test_resnet_batch.py     #   ResNet 批量性能测试
│   └── test_resnet_batch_multi.py   # ResNet 多请求合并 (dynamic batching) 测试
│
├── model_repo/                  # 模型仓库 (4 个模型)
│   ├── resnet50/                #   ResNet50  · migraphx 后端 · FP16 固定 batch 128
│   ├── resnet50_onnx/           #   ResNet50  · onnx 后端    · FP16 动态 batch + dynamic_batching
│   ├── yolov5s/                 #   YOLOv5s   · onnx 后端    · FP16 动态 batch
│   └── yolov5s_mx/              #   YOLOv5s   · migraphx 后端 · FP32
│
└── samples/
    ├── bus.jpg                  # 示例图片
    ├── bus_input.bin            # YOLO 输入 (1,3,640,640) FP16
    └── resnet_input.bin         # ResNet 输入 (1,3,224,224) FP16
```

> 目录约定：**C++ 源码在 `src/`，Python 脚本在 `python/`**；脚本/配置在仓库根；C++ 可执行产物编译到仓库根（已被 `.gitignore` 忽略）。

---

## 二、前置条件

- 海光 HCU + DTK 运行环境（`/opt/dtk`）
- Docker（`run_all.sh` 使用镜像 `dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12`）
- 可访问 `developer.sourcefind.cn`（拉源码/模型）与 `pypi.tuna.tsinghua.edu.cn`（装 Python 包）
- **git 凭据**（容器内 `/root/.git-credentials`，容器重建后需重配）：
  ```bash
  echo "https://<user>:<token>@developer.sourcefind.cn" > /root/.git-credentials
  git config --global credential.helper store
  ```

`build.sh` 会自动安装：migraphx（`.run`）、onnxruntime（`.whl`）、cmake 3.31.8、boost 1.82、rapidjson、protobuf 等，并编译 core / backend / onnx_backend / migraphx_backend / server。

---

## 三、快速开始

### GitHub Actions CI

CI 先从共享镜像包准备同仓库的私有 GHCR 镜像，再使用 K8s + ARC 调度编译和测试作业。
具体节点由标签对应的 runner 池自动选择，缺失镜像时由容器运行时按需拉取：

| 作业 | runs-on | 工作内容 |
| --- | --- | --- |
| prepare-image | `hygon-build` | 校验共享包，发布或复用同仓库的私有 GHCR 镜像，输出固定 digest |
| build | `hygon-build` | 编译 server、ONNX Runtime、MIGraphX 和 PyTorch backend，发布安装产物 |
| test | `hygon-bw1100-1card` | 使用 build 产物执行 HTTP/gRPC、C++/Python 和模型推理测试 |

现有四个 GPU 模型都指定 `gpus: [0]`，PyTorch 加法模型使用 CPU，因此只申请一张
HCU 卡。8 个并发请求在同一张卡执行，不要求 8 张卡。batch 128 仍需足够显存，
实际是否跑通以单卡作业结果为准。

使用前，在仓库 Settings → Secrets and variables → Actions → Variables 创建
`CI_IMAGE`，填写包含完整 HCU 构建和运行依赖的镜像地址，建议固定 digest。
手动 Run workflow 也可通过 `ci_image` 输入覆盖镜像；build 和 test 始终使用同一配置。
没有提供镜像时不会回退到之前缺少 Torch 的 DTK 基础镜像。

每次 CI 触发后，`prepare-image` 在 `hygon-build` 分配的节点上使用固定 digest 的
公开 Skopeo 基础镜像读取共享包。首次发布先核对压缩包 SHA-256 和源镜像 ID，再上传到
`ghcr.io/hygon-ai/triton-inference-server-test-ci` 的私有 package；已有匹配镜像时跳过上传。
已有 package 必须是 private，否则直接失败。发布后再核对镜像 ID，并将不可变的
`@sha256:...` 镜像引用传给 build 和 test，两个作业使用相同 digest。

镜像发布使用该仓库自带的 `GITHUB_TOKEN`，仅准备作业声明 `packages: write`；
编译、测试仅使用 `packages: read`，通过 `container.credentials` 自动提供拉取凭据。
不需要个人 token、K8s kubeconfig、节点列表或 SSH 权限，不修改 Pod 模板，也不安装 cron。
任何符合原有 runner 池调度条件的节点，都由 K8s 在启动任务容器时检查缓存并按需拉取镜像，
无需在 CI 中预先枚举节点，也不复用手动启动的容器。

共享源包仍保留在：

```text
/ci_public/ci-images/triton-inference-server-test-ci/20261111-aicc-g2f732c
```

目录须包含 `image.tar.gz`、`image.tar.gz.sha256`、`image-tag`、`image-id`。
Actions variable `CI_IMAGE_ARCHIVE_DIR` 可覆盖该目录。`CI_IMAGE` / `ci_image` 表示共享源包
对应的镜像 tag，须与 `image-tag` 匹配；实际作业镜像由前置任务输出的 GHCR digest 决定。
`CI_IMAGE_NODES`、`CI_KUBECONFIG_PATH` 和 `CI_K8S_READONLY_KUBECONFIG` 均不再使用。

首次发布需传输镜像层，耗时取决于网络；后续复用已发布的镜像，不会每轮重新上传。
节点需要能拉取公开的 Skopeo 基础镜像及私有 GHCR 镜像。私有 package 创建或拉取受到组织
策略阻止时，CI 明确失败，不回退到未检查的本地镜像。

当前脚本对应 Python 3.12、HCU/HIP Torch 2.7.1、匹配的 HCU ONNX Runtime 1.19.2、
`numpy<2`、DTK/MIGraphX，以及 CMake、Boost、protobuf、rapidjson 等构建依赖。
不能只替换成任意版本的 Torch 镜像；当前 ONNX Runtime wheel 对应 DTK 26.04 / Torch 2.7.1。
CI 在构建前检查框架类型和版本，跳过 `build.sh` 中自动安装系统/框架依赖的步骤，
并约束后续 pip 安装，避免自动从 PyPI 替换成 CUDA Torch。
若构建节点无 GPU、编译器需要显式架构，可设置 `CI_HIP_ARCHITECTURES` 为测试卡
对应的 HIP 架构（或由镜像提供 `AMDGPU_TARGETS`），不要沿用其他型号的架构值。

ARC 平台需在两类作业容器内挂载同一份 `/ci_public` 共享存储，并为单卡作业分配
HCU 设备；本流程的私有 GHCR 拉取凭据由 workflow 自动提供。产物和日志位于：

```text
/ci_public/triton-inference-server-test/runs/<run-id>-<build-attempt>/
```

测试作业通过 build 输出取得此次安装目录，核对测试提交和文件校验和，并安装同次
构建的 Python bindings。两端使用相同绝对安装路径，避免原生库引用编译节点临时目录。
仅重跑失败的测试作业仍使用此前成功的 build 产物；平台需保留这些产物至重跑完成，
历史目录由平台统一安排清理，不能视作永久缓存。

测试仓库直接 checkout 本次提交，不复制源码，也不执行 `git pull`。依赖仓库由
`build.sh` 拉取；若依赖仓库为私有，需要额外设置能只读访问这些仓库的
`CI_REPO_TOKEN` Actions secret，当前仓库的默认 `GITHUB_TOKEN` 不能代替跨仓库授权。
任一编译或测试失败都会使 CI 失败，各服务测试结束时清理自己启动的进程。
以下 `run_all.sh` 的 Docker 模式保留供手动使用，与 ARC 作业分开。

### 方式 A：宿主机一键编排（推荐）
```bash
# 参数: <宿主机挂载点> <容器内 WORK_BASE>
./run_all.sh /public/home/luoxiao/ci /home/ci

# 可选模式:
./run_all.sh --test-only /public/home/luoxiao/ci /home/ci   # 跳过 build.sh (已构建)
./run_all.sh --force     /public/home/luoxiao/ci /home/ci   # 强制完整重建
```

### 方式 B：容器内分步
```bash
docker exec -it triton_inference_server_ci bash
export WORK_BASE=/home/ci

# 1) 完整构建 (装依赖 + 编译 + 装 Python 包) —— 只需一次
bash ${WORK_BASE}/triton-inference-server-test/build.sh ${WORK_BASE} HYGON-AI

# 2) 跑测试 (启动 server + HTTP/gRPC + C++/Python 客户端)
cd ${WORK_BASE}/triton-inference-server-test
bash run_test.sh ${WORK_BASE}
```

---

## 四、脚本说明

| 脚本 | 位置 | 职责 |
|---|---|---|
| `build.sh` | 容器内 | **构建 + 安装**：装系统/Python 依赖、编译 core/backend(s)/server，末尾安装 Python 测试依赖（`tritonserver` wheel + `tritonclient[grpc]`） |
| `run_test.sh` | 容器内 | **跑测试**：`run_server.sh` → 编译 `src/*.cc` → YOLO/ResNet 各测试。**不含任何 `pip install`** |
| `run_server.sh` | 容器内 | **服务端测试**：启动 HTTP=8000 / gRPC=8001 的 `tritonserver`，跑 `python/test_yolo_http.py`/`python/test_yolo_grpc.py`，结束后 kill 进程 |
| `run_all.sh` | 宿主机 | **一键编排**：clone test repo → 创建/启动容器 → `docker exec build.sh` → `docker exec run_test.sh` → 健康检查 |
| `env.sh` | 容器内 | 环境变量：`LD_LIBRARY_PATH`（`opt/triton-server/core/lib` 等）、`CMAKE_PREFIX_PATH` |
| `docker_create.sh` | 宿主机 | 单独创建容器（调试） |

> **安装与运行分离**：依赖安装统一在 `build.sh`；`run_test.sh` / `run_server.sh` 只负责运行测试。

---

## 五、模型仓库（model_repo）

| 模型 | 后端 | 精度 | batch | 说明 |
|---|---|---|---|---|
| `resnet50` | migraphx | FP16 | **固定 128** | 图像分类（1000 类） |
| `resnet50_onnx` | onnx | FP16 | 动态 + dynamic_batching | 图像分类，可接任意 batch |
| `yolov5s` | onnx | FP16 | 动态 | 目标检测，输出 `[B,25200,85]` |
| `yolov5s_mx` | migraphx | FP32 | 固定 128 | 目标检测 |

模型规格：
- **ResNet50**：输入 `input_1.1 [B,3,224,224]`，输出 `softmax_tensor:0 [B,1000]`
- **YOLOv5s**：输入 `images [B,3,640,640]`，输出 `output0 [B,25200,85]`

---

## 六、测试用例

### C++ C-API 客户端（源码 `src/`，产物在根）
```bash
# 编译 (run_test.sh 会自动做)
g++ -O2 src/test_backend_infer.cc -I${WORK_BASE}/opt/triton-server/core/include \
    -L${WORK_BASE}/opt/triton-server/core/lib -ltritonserver -lpthread -o test_backend_infer
g++ -O2 src/test_yolo.cc -I${WORK_BASE}/opt/triton-server/core/include \
    -L${WORK_BASE}/opt/triton-server/core/lib -L/opt/dtk/lib -Wl,-rpath-link,/opt/dtk/lib \
    -ltritonserver -lpthread -o test_yolo

# 运行 (ResNet, FP16 batch 128)
./test_backend_infer resnet50       ${WORK_BASE}/opt/triton-server/backends/ ./model_repo/
./test_backend_infer resnet50_onnx  ${WORK_BASE}/opt/triton-server/backends/ ./model_repo/
# 运行 (YOLO)
./test_yolo yolov5s ${WORK_BASE}/opt/triton-server/backends/ ./model_repo/ samples/bus_input.bin
```

### Python in-process（`tritonserver` 绑定，脚本在 `python/`）
```bash
python3 python/test_yolo.py ./model_repo ${WORK_BASE}/opt/triton-server/backends samples/bus_input.bin
python3 python/test_yolo_batch.py   ./model_repo ${WORK_BASE}/opt/triton-server/backends samples/bus_input.bin 128 10 yolov5s fp16
python3 python/test_resnet_batch.py ./model_repo ${WORK_BASE}/opt/triton-server/backends samples/resnet_input.bin 128 10 resnet50_onnx fp16
python3 python/test_resnet_batch_multi.py ./model_repo ${WORK_BASE}/opt/triton-server/backends samples/resnet_input.bin 128 5 resnet50_onnx fp16 8
```

### HTTP / gRPC 端到端
```bash
python3 python/test_yolo_http.py http://127.0.0.1:8000 samples/bus_input.bin
python3 python/test_yolo_grpc.py 127.0.0.1:8001        samples/bus_input.bin
```

### 关于 `python/triton_compat.py`（重要）
Python `import tritonserver` 时，C 扩展默认以 `RTLD_LOCAL` 加载，导致 `libtritonserver.so` 的 `TRITONBACKEND_*` 符号**不进入全局符号表**；随后 in-process dlopen backend 时会报：
```
undefined symbol: TRITONBACKEND_ModelInstanceModel
```
`triton_compat.enable_global_symbols()`（内部 `sys.setdlopenflags(RTLD_GLOBAL)`）解决该问题，**替代 `LD_PRELOAD=libtritonserver.so`**。所有 in-process 脚本在 `import tritonserver` 前调用：
```python
from triton_compat import enable_global_symbols
enable_global_symbols()
import tritonserver
```
> 因脚本与 `triton_compat.py` 同在 `python/`，运行时 `sys.path[0]` 即 `python/`，可直接 import。

---

## 七、输入数据

- `samples/bus_input.bin`：YOLO 输入 `(1,3,640,640)` FP16，像素归一化 `[0,1]`
- `samples/resnet_input.bin`：ResNet 输入 `(1,3,224,224)` FP16
- 自制 YOLO 输入：
  ```bash
  python3 python/build_yolo_input.py <image.jpg> [output.bin]
  ```

---

## 八、性能参考（batch=128）

| 模型 | 后端 | 精度 | 吞吐量 |
|---|---|---|---|
| ResNet50 | onnx | FP32 | 1694 samples/s |
| ResNet50 | onnx | FP16 | 3306 samples/s |
| ResNet50 | migraphx | FP32 | 2342 samples/s |
| **ResNet50** | **migraphx** | **FP16** | **4967 samples/s** |
| YOLOv5s | onnx | FP16 | ~520 samples/s |

---

## 九、已知问题与注意事项

1. **migraphx 不支持动态 batch**
   HCU 版 migraphx 编译期固化 shape，动态 batch 模型会被编译成 batch=1。migraphx 模型需用**固定 batch**（`resnet50`/`yolov5s_mx` 为固定 128）。

2. **migraphx + FP16 + 动态 shape 算子**
   YOLO 的 Detect 头含 `Shape/Gather/Range/Expand/Where` 等动态算子，migraphx 对 FP16 输出的 shape/buffer 处理有 bug（输出 buffer 分配异常）。**纯静态算子图（如 ResNet）的 migraphx FP16 正常**。

3. **请求 batch 与 `max_batch_size`**
   - `max_batch_size` 是**硬上限**：请求 batch > 上限时 Triton **入口拒绝**（不执行）
   - 固定 batch 模型：请求 batch **必须等于**模型 batch，否则 backend 报 shape 错误
   - `dynamic_batching` 的 `preferred_batch_size`/`max_queue_delay_microseconds` 会为"凑批"引入延迟（batch 不足时会白等），按负载调整

4. **容器重建后**
   - Python 包（tritonserver / tritonclient / onnx 等）丢失 → 重新跑 `build.sh`
   - `/root/.git-credentials` 丢失 → 重配 git 凭据

5. **`onnx` Python 模块**
   仅模型转换脚本需要（`pip install onnx`），跑推理测试不需要。

---

## 十、常见问题排查

| 现象 | 原因 / 处理 |
|---|---|
| `failed to load all models` | 看 CLI 日志定位是哪个模型；常见为符号/schema 问题或加载超时（migraphx 编译慢，可达 50s+） |
| `undefined symbol: TRITONBACKEND_ModelInstanceModel` | in-process 缺全局符号 → 用 `python/triton_compat.py`（勿依赖 `LD_PRELOAD`） |
| `No module named 'tritonclient'` | 重跑 `build.sh` 安装 `tritonclient[grpc]` |
| `No module named 'onnx'` | `pip install onnx`（仅转换脚本需要） |
| `inference request batch-size must be <= N` | 请求 batch 超过 `max_batch_size`，Triton 拒绝 |
| `Incorrect shape ... should be {128,...}` | 固定 batch 模型收到了非 128 的请求 |
| `No module named 'triton_compat'` | 需在 `python/` 目录下运行脚本（或在脚本开头把 `python/` 加入 `sys.path`） |
| migraphx 加载慢 | 编译期图优化，属正常（ResNet ~28s，YOLO ~50s） |

---

## 许可证

本仓库采用 BSD-3-Clause 许可证，完整文本见 [LICENSE](LICENSE)。

## 第三方来源

本仓库当前不包含第三方源码；`model_repo/` 下的 ONNX 模型与 `samples/` 下的测试图片属于测试数据。
后续如引入第三方源码，必须在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中登记项目、仓库、
固定版本或 Commit、版权、许可证、本地路径与 HYGON 修改说明。
