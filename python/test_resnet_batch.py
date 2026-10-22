#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""test_resnet_batch.py - ResNet50 批量推理性能测试 (in-process)

用法: python3 test_resnet_batch.py [model_repo] [backend_dir] [input.bin] [batch] [iters] [model_name] [fp16|fp32]
"""
import sys, time, gc
import numpy as np
from triton_compat import enable_global_symbols
enable_global_symbols()   # 让 libtritonserver 符号全局可见 (替代 LD_PRELOAD)
import tritonserver

def main():
    model_repo = sys.argv[1] if len(sys.argv) > 1 else "./model_repo"
    backend_dir = (sys.argv[2] if len(sys.argv) > 2
                   else "/home/ci/opt/triton-server/backends")
    input_bin = sys.argv[3] if len(sys.argv) > 3 else "samples/resnet_input.bin"
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    model_name = sys.argv[6] if len(sys.argv) > 6 else "resnet50"
    dtype = np.float32 if (len(sys.argv) > 7 and sys.argv[7] == "fp32") else np.float16  # 默认 FP16

    print(f"启动 Triton Server (model={model_name}, batch={batch}, iters={iters})...")
    server = tritonserver.Server(
        model_repository=model_repo, backend_directory=backend_dir,
        log_verbose=0, exit_timeout=5,
    )
    server.start(wait_until_ready=True)
    try:
        model = server.model(model_name)

        # 输入: 单图复制成 (batch,3,224,224)
        base = np.fromfile(input_bin, dtype=np.float16).reshape(1, 3, 224, 224).astype(dtype)
        arr = np.repeat(base, batch, axis=0)
        in_bytes = arr.nbytes / 1024 / 1024
        print(f"输入: {arr.shape} {arr.dtype} ({in_bytes:.0f} MB)")
        out_bytes = batch * 1000 * arr.dtype.itemsize / 1024 / 1024
        print(f"输出估算: batch*1000 ≈ {out_bytes:.1f} MB")

        # 热身 (冷启动不计入)
        list(model.infer(inputs={"input_1.1": arr}))
        print("热身完成")

        # 性能测量
        t0 = time.time()
        for i in range(iters):
            resp = list(model.infer(inputs={"input_1.1": arr}))
            out = np.from_dlpack(resp[0].outputs["softmax_tensor:0"])
            assert out.shape[0] == batch, f"输出 batch 异常: {out.shape}"
        dt = time.time() - t0

        total = batch * iters
        throughput = total / dt
        avg_ms = dt / iters * 1000
        print(f"\n>>>性能结果 (batch={batch}, iters={iters})")
        print(f"  总样本: {total}")
        print(f"  总耗时: {dt:.3f} s")
        print(f"  平均延迟/请求: {avg_ms:.1f} ms")
        print(f"  吞吐量: {throughput:.1f} samples/s ({throughput/1000:.2f} K samples/s)")
        print(f"  每 batch 耗时: {avg_ms:.1f} ms")

        # 后处理验证 (top5 类别)
        pred = out[0].astype(np.float32)
        top5 = np.argsort(pred)[::-1][:5]
        print(f"  末批次第1图 top5: {top5.tolist()} (验证输出有效)")

        print(">>>success inference:resnet")
    finally:
        try:
            del out, resp
        except Exception:
            pass
        gc.collect()
        server.stop()
        print("Server 已停止")

if __name__ == "__main__":
    main()
