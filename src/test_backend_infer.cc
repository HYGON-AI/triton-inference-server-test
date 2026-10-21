// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
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

// 用于跨线程同步推理完成状态的上下文结构体
struct InferContext {
  std::mutex mtx;
  std::condition_variable cv;
  bool completed = false;
  std::vector<uint16_t> output_buffer;  // FP16
};

// 1. 响应输出内存分配回调函数
TRITONSERVER_Error* ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id) {
  
  auto* ctx = reinterpret_cast<InferContext*>(userp);
  ctx->output_buffer.resize(byte_size / sizeof(uint16_t));  // FP16

  *buffer = ctx->output_buffer.data();
  *buffer_userp = nullptr;
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = 0;

  return nullptr;
}

// 释放回调（使用系统内存，无需额外动作）
TRITONSERVER_Error* ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer,
    void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id) {
  return nullptr;
}

// 2. 推理执行完成回调函数
void InferComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags,
    void* userp) {
  auto* ctx = reinterpret_cast<InferContext*>(userp);

  // 检查推理过程是否有错
  TRITONSERVER_Error* response_err = TRITONSERVER_InferenceResponseError(response);
  if (response_err != nullptr) {
    std::cerr << "推理失败: " << TRITONSERVER_ErrorMessage(response_err) << std::endl;
    TRITONSERVER_ErrorDelete(response_err);
  }

  // 释放响应对象
  if (response != nullptr) {
    TRITONSERVER_InferenceResponseDelete(response);
  }

  // 通知主线程
  std::lock_guard<std::mutex> lock(ctx->mtx);
  ctx->completed = true;
  ctx->cv.notify_one();
}

int main(int argc,char** args) {
  std::string model_name = "resnet50";
  const int64_t model_version = 1;
  char* model_repo="./model_repo";
  char* backend_path="/home/ci/opt/triton-server/backends";
  if(argc>1){
    model_name=args[1];
  }
  if(argc>2){
    backend_path=args[2];
  }
  if(argc>3){
    model_repo=args[3];
  }

  std::cout<<"model_name:"<<model_name<<" backend_path:"<<backend_path<<" model_repo:"<<model_repo<<std::endl;

  // -------------------------------------------------------------
  // Step 1: 初始化 Triton Server
  // -------------------------------------------------------------
  TRITONSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "创建 ServerOptions");
  
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 1), "设置 LogVerbose");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogInfo(server_options, true), "设置 LogInfo");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogWarn(server_options, true), "设置 LogWarn");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogError(server_options, true), "设置 LogError");

  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                  server_options, model_repo), "设置 Model Repository");
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(
                  server_options, backend_path), "设置 Backend 路径");

  TRITONSERVER_Server* server = nullptr;
  std::cout << "正在启动 Triton Core 并加载模型..." << std::endl;
  FAIL_IF_ERR(TRITONSERVER_ServerNew(&server, server_options), "创建 Server 失败");

  bool is_ready = false;
  FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(server, model_name.c_str(), model_version, &is_ready),
              "检查模型状态");
  if (!is_ready) {
    std::cerr << "模型未准备就绪！" << std::endl;
    return 1;
  }
  std::cout << "模型加载成功，开始构建输入数据并执行推理..." << std::endl;

  // -------------------------------------------------------------
  // Step 2: 准备虚拟输入数据 (1, 3, 224, 224) FP16
  // -------------------------------------------------------------
  std::vector<int64_t> input_shape = {128, 3, 224, 224};
  size_t input_elements = 128 * 3 * 224 * 224;
  std::vector<uint16_t> input_data(input_elements, 0x3800); // FP16(0.5) 填充测试数值
  size_t input_byte_size = input_data.size() * sizeof(uint16_t);

  // -------------------------------------------------------------
  // Step 3: 创建 Allocator 与 InferenceRequest
  // -------------------------------------------------------------
  InferContext ctx;

  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(
                  &allocator, ResponseAlloc, ResponseRelease, nullptr), "创建 Allocator");

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(
                  &irequest, server, model_name.c_str(), model_version), "创建 InferenceRequest");

  // 添加输入张量描述
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                  irequest, "input_1.1", TRITONSERVER_TYPE_FP16,
                  input_shape.data(), input_shape.size()), "添加输入参数元数据");

  // 绑定输入数据内存
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                  irequest, "input_1.1", input_data.data(), input_byte_size,
                  TRITONSERVER_MEMORY_CPU, 0), "绑定输入数据内容");

  // 声明请求的输出张量
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
                  irequest, "softmax_tensor:0"), "添加请求输出");

  // 设置完成回调与响应分配器
  FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
                  irequest, [](TRITONSERVER_InferenceRequest* req, const uint32_t flags, void* userp) {
                    TRITONSERVER_InferenceRequestDelete(req);
                  }, nullptr), "设置请求释放回调");

  FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                  irequest, allocator, &ctx, InferComplete, &ctx), "设置响应分配器及回调");

  // -------------------------------------------------------------
  // Step 4: 异步派发推理并同步等待结果
  // -------------------------------------------------------------
  FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(
                  server, irequest, nullptr), "发送异步推理请求");

  {
    std::unique_lock<std::mutex> lock(ctx.mtx);
    ctx.cv.wait(lock, [&ctx] { return ctx.completed; });
  }

  // -------------------------------------------------------------
  // Step 5: 验证输出结果
  // -------------------------------------------------------------
  std::cout << "\n>>>Success Inference:"<<model_name << std::endl;
  std::cout << "输出张量元素总数: " << ctx.output_buffer.size() << " (预期 128000)" << std::endl;
  std::cout << "前 5 个输出值（FP16 位模式）: ";
  for (size_t i = 0; i < std::min<size_t>(5, ctx.output_buffer.size()); ++i) {
    std::cout << ctx.output_buffer[i] << " ";
  }
  std::cout << std::endl;

  // -------------------------------------------------------------
  // Step 6: 清理全局资源
  // -------------------------------------------------------------
  TRITONSERVER_ResponseAllocatorDelete(allocator);
  TRITONSERVER_ServerDelete(server);
  TRITONSERVER_ServerOptionsDelete(server_options);

  return 0;
}
