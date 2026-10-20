// test_yolo.cc
// 目标检测 (YOLOv5s) Triton 推理样例客户端
// 用法: ./test_yolo [model_name] [backend_path] [model_repo]
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include "triton/core/tritonserver.h"

#define FAIL_IF_ERR(X, MSG)                                          \
  do {                                                               \
    TRITONSERVER_Error* err__ = (X);                                 \
    if (err__ != nullptr) {                                          \
      std::cerr << "错误 [" << MSG << "]: "                         \
                << TRITONSERVER_ErrorMessage(err__) << std::endl;    \
      TRITONSERVER_ErrorDelete(err__);                               \
      exit(1);                                                       \
    }                                                                \
  } while (false)

// FP16 -> FP32 转换
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t man  = h & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) { f = sign; }
    else {
      exp = 127 - 15 + 1;
      while ((man & 0x0400u) == 0) { man <<= 1; exp--; }
      man &= 0x03FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);  // inf / nan
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

// 检测框结构
struct Box { float x1, y1, x2, y2; float score; int cls; };
static float box_iou(const Box& a, const Box& b) {
  float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
  float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  float inter = iw * ih;
  float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0 ? inter / uni : 0.0f;
}

static std::vector<Box> nms(std::vector<Box> boxes, float iou_thr) {
  std::sort(boxes.begin(), boxes.end(),
            [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> keep;
  std::vector<bool> removed(boxes.size(), false);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (removed[i]) continue;
    keep.push_back(boxes[i]);
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (!removed[j] && boxes[j].cls == boxes[i].cls &&
          box_iou(boxes[i], boxes[j]) > iou_thr)
        removed[j] = true;
    }
  }
  return keep;
}

// 推理上下文
struct InferContext {
  std::mutex mtx;
  std::condition_variable cv;
  bool completed = false;
  std::vector<char> output_buffer;   // 输出原始字节
  std::vector<int64_t> output_shape; // 输出形状
};

TRITONSERVER_Error* ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id) {
  auto* ctx = reinterpret_cast<InferContext*>(userp);
  ctx->output_buffer.resize(byte_size);
  *buffer = ctx->output_buffer.data();
  *buffer_userp = nullptr;
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = 0;
  return nullptr;
}

TRITONSERVER_Error* ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer,
    void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id) {
  return nullptr;
}

void InferComplete(TRITONSERVER_InferenceResponse* response,
                   const uint32_t flags, void* userp) {
  auto* ctx = reinterpret_cast<InferContext*>(userp);
  TRITONSERVER_Error* response_err =
      TRITONSERVER_InferenceResponseError(response);
  if (response_err != nullptr) {
    std::cerr << "推理失败: " << TRITONSERVER_ErrorMessage(response_err)
              << std::endl;
    TRITONSERVER_ErrorDelete(response_err);
  }
  // 释放响应对象（不在此处调用 TRITONSERVER_InferenceResponseOutput，
  // 输出数据已由 ResponseAlloc 写入 ctx->output_buffer）
  if (response != nullptr) {
    TRITONSERVER_InferenceResponseDelete(response);
  }
  std::lock_guard<std::mutex> lock(ctx->mtx);
  ctx->completed = true;
  ctx->cv.notify_one();
}

int main(int argc, char** args) {
  std::string model_name = "yolov5s";
  const int64_t model_version = 1;
  char* model_repo = const_cast<char*>("./model_repo");
  char* backend_path = const_cast<char*>("/home/ci/opt/triton-server/backends");
  char* input_file = nullptr;  // 可选：真实图片预处理后的 FP16 输入文件
  if (argc > 1) model_name = args[1];
  if (argc > 2) backend_path = args[2];
  if (argc > 3) model_repo = args[3];
  if (argc > 4) input_file = args[4];

  std::cout << "model_name:" << model_name << " backend_path:" << backend_path
            << " model_repo:" << model_repo << std::endl;

  // Step 1: 初始化 Triton
  TRITONSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "创建 ServerOptions");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 1), "LogVerbose");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogInfo(server_options, true), "LogInfo");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogWarn(server_options, true), "LogWarn");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogError(server_options, true), "LogError");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, model_repo), "ModelRepository");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, backend_path), "BackendDirectory");

  TRITONSERVER_Server* server = nullptr;
  std::cout << "正在启动 Triton Core 并加载 yolov5s..." << std::endl;
  FAIL_IF_ERR(TRITONSERVER_ServerNew(&server, server_options), "创建 Server 失败");

  bool is_ready = false;
  FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(server, model_name.c_str(), model_version, &is_ready), "检查模型状态");
  if (!is_ready) {
    std::cerr << "模型未准备就绪！" << std::endl;
    return 1;
  }
  std::cout << "模型加载成功，开始构建输入并推理..." << std::endl;

  // Step 2: 构造 FP16 图像输入 (1, 3, 640, 640)
  std::vector<int64_t> input_shape = {1, 3, 640, 640};
  size_t input_elements = 1 * 3 * 640 * 640;
  std::vector<uint16_t> input_fp16(input_elements);
  if (input_file != nullptr) {
    // 从文件加载真实图片预处理后的 FP16 原始数据
    FILE* f = fopen(input_file, "rb");
    if (f == nullptr) {
      std::cerr << "无法打开输入文件: " << input_file << std::endl;
      return 1;
    }
    size_t rd = fread(input_fp16.data(), sizeof(uint16_t), input_elements, f);
    fclose(f);
    if (rd != input_elements) {
      std::cerr << "输入文件大小不匹配，期望 " << input_elements * 2
                << " 字节，实际读取 " << rd * 2 << std::endl;
      return 1;
    }
    std::cout << "已加载真实图像输入: " << input_file << std::endl;
  } else {
    for (auto& v : input_fp16) v = 0x3800;  // 0.5f 的 FP16 编码（灰色虚拟图）
    std::cout << "未提供输入文件，使用灰色虚拟图" << std::endl;
  }
  size_t input_byte_size = input_fp16.size() * sizeof(uint16_t);

  // Step 3: 创建请求
  InferContext ctx;
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease, nullptr), "Allocator");

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&irequest, server, model_name.c_str(), model_version), "Request");

  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, "images", TRITONSERVER_TYPE_FP16, input_shape.data(), input_shape.size()), "AddInput");
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, "images", input_fp16.data(), input_byte_size, TRITONSERVER_MEMORY_CPU, 0), "AppendInput");
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, "output0"), "AddOutput");

  FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest,
      [](TRITONSERVER_InferenceRequest* req, const uint32_t flags, void* userp) {
        TRITONSERVER_InferenceRequestDelete(req);
      }, nullptr), "SetReleaseCallback");
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(irequest, allocator, &ctx, InferComplete, &ctx), "SetResponseCallback");

  // Step 4: 异步推理并等待
  FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(server, irequest, nullptr), "异步推理");
  {
    std::unique_lock<std::mutex> lock(ctx.mtx);
    ctx.cv.wait(lock, [&ctx] { return ctx.completed; });
  }

  // Step 5: 后处理 YOLOv5 输出
  std::cout << "\n>>>目标检测推理完成" << std::endl;
  std::cout << "输出字节数: " << ctx.output_buffer.size() << std::endl;

  const uint16_t* raw = reinterpret_cast<const uint16_t*>(ctx.output_buffer.data());
  // yolov5s 输出固定为 [1, 25200, 85] FP16，从 buffer 大小推导
  int64_t cols = 85;
  int64_t numel = (int64_t)ctx.output_buffer.size() / 2;  // uint16 元素数
  int64_t rows = numel / cols;
  std::cout << "输出元素数(rows=" << rows << ", cols=" << cols << ")" << std::endl;

  const float conf_thr = 0.25f, iou_thr = 0.45f;
  std::vector<Box> dets;
  for (int64_t i = 0; i + cols <= numel; i += cols) {
    if (i + cols > numel) break;
    float cx = fp16_to_fp32(raw[i]);
    float cy = fp16_to_fp32(raw[i + 1]);
    float w  = fp16_to_fp32(raw[i + 2]);
    float h  = fp16_to_fp32(raw[i + 3]);
    float obj = fp16_to_fp32(raw[i + 4]);
    int best_cls = 0; float best_score = -1.0f;
    for (int c = 0; c < 80; ++c) {
      float s = fp16_to_fp32(raw[i + 5 + c]);
      if (s > best_score) { best_score = s; best_cls = c; }
    }
    float score = obj * best_score;
    if (score > conf_thr) {
      dets.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, best_cls});
    }
  }
  std::cout << "阈值过滤后候选框数: " << dets.size() << std::endl;
  auto final_dets = nms(dets, iou_thr);
  std::cout << "NMS 后检测结果数: " << final_dets.size() << std::endl;
  const char* coco_names[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"};
  for (size_t k = 0; k < final_dets.size() && k < 10; ++k) {
    const Box& b = final_dets[k];
    std::cout << "  [" << k << "] class=" << (b.cls >= 0 && b.cls < 80 ? coco_names[b.cls] : "?")
              << "(" << b.cls << ") score=" << b.score
              << " box=(" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2 << ")"
              << std::endl;
  }

  if(final_dets.size()==6){
     std::cout << "\n>>>Success Inference:"<<model_name << std::endl;
  }else{
     std::cout << "\n>>>Fail Inference:"<<model_name << std::endl;
  }

  // Step 6: 清理
  TRITONSERVER_ResponseAllocatorDelete(allocator);
  TRITONSERVER_ServerDelete(server);
  TRITONSERVER_ServerOptionsDelete(server_options);
  return 0;
}
