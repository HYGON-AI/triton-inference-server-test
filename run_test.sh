export WORK_BASE=$1
export SCRIPT_PATH=$(dirname $(realpath "$0") )
export env_path=${SCRIPT_PATH}/env.sh

echo "env_path: $env_path"
. $env_path

#测试运行
#git clone https://developer.sourcefind.cn/codes/OpenDAS/triton-inference-server-test.git

cd $WORK_BASE/triton-inference-server-test

./run_server.sh $WORK_BASE

g++ -O2 src/test_backend_infer.cc  -I${WORK_BASE}/opt/triton-server/include -L${WORK_BASE}/opt/triton-server/lib -ltritonserver -lpthread -o test_backend_infer
./test_backend_infer resnet50 ${WORK_BASE}/opt/triton-server/backends/ ${SCRIPT_PATH}/model_repo/

./test_backend_infer resnet50_onnx ${WORK_BASE}/opt/triton-server/backends/ ${SCRIPT_PATH}/model_repo/

# ============ 目标检测样例 (YOLOv5s) ============
# 编译客户端（需要链接 DTK 库以解析 libtritonserver 的 HIP 依赖）
g++ -O2 src/test_yolo.cc \
  -I${WORK_BASE}/opt/triton-server/include \
  -L${WORK_BASE}/opt/triton-server/lib -L/opt/dtk/lib \
  -Wl,-rpath-link,/opt/dtk/lib \
  -ltritonserver -lpthread -o test_yolo

# 运行目标检测推理（第 4 个参数为真实图片预处理后的 FP16 输入，
# 预处理方法见 python/build_yolo_input.py；无参则用灰色虚拟图）
./test_yolo yolov5s ${WORK_BASE}/opt/triton-server/backends/ ${SCRIPT_PATH}/model_repo/ ${SCRIPT_PATH}/samples/bus_input.bin

# ============ 目标检测 Python 客户端 (tritonserver Python API) ============
# 运行 Python 目标检测测试 (Python 依赖由 build.sh 安装)
python3 -u python/test_yolo.py ${SCRIPT_PATH}/model_repo ${WORK_BASE}/opt/triton-server/backends ${SCRIPT_PATH}/samples/bus_input.bin

# ============ 图像分类批量性能 (ResNet50, batch 128, FP16) ============
# onnx backend (dynamic batch): 单请求 batch 128
python3 -u python/test_resnet_batch.py ${SCRIPT_PATH}/model_repo ${WORK_BASE}/opt/triton-server/backends ${SCRIPT_PATH}/samples/resnet_input.bin 128 10 resnet50_onnx fp16

# migraphx backend (固定 batch 128)
python3 -u python/test_resnet_batch.py ${SCRIPT_PATH}/model_repo ${WORK_BASE}/opt/triton-server/backends ${SCRIPT_PATH}/samples/resnet_input.bin 128 10 resnet50 fp16

# ============ 多请求合并 (dynamic batching: 8 x 16 -> 128) ============
# 并发发送 8 个请求 (每请求 16), 验证 dynamic batcher 合并为 1 次执行;
# backend 会打印 [BATCH] execute: request_count=8 total_batch_size=128
python3 -u python/test_resnet_batch_multi.py ${SCRIPT_PATH}/model_repo ${WORK_BASE}/opt/triton-server/backends ${SCRIPT_PATH}/samples/resnet_input.bin 128 5 resnet50_onnx fp16 8

# PyTorch backend case (pytorch_add)
./run_pytorch_backend.sh $WORK_BASE
