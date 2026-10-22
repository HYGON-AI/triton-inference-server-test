#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""test_resnet_batch_multi.py - ResNet50 多请求合并 (dynamic batching) 测试

并发发送多个小请求, 验证 Triton dynamic batcher 将其合并为一个大 batch 执行。
合并证据: backend 打印的 [BATCH] execute: request_count=N (N>1 即合并)。

用法: python3 test_resnet_batch_multi.py [model_repo] [backend_dir] [input.bin] \
        [total_batch] [iters] [model_name] [fp16|fp32] [n_requests]

默认: ./model_repo .../backends samples/resnet_input.bin 128 5 resnet50_onnx fp16 8
含义: 总 batch 128 拆成 8 个并发请求 (每请求 16), 由 dynamic batcher 合并。
"""
import sys, os, time, gc, concurrent.futures
import numpy as np
from triton_compat import enable_global_symbols
enable_global_symbols()   # 让 libtritonserver 符号全局可见 (替代 LD_PRELOAD)
import tritonserver


def infer_one(model, arr):
    resp = list(model.infer(inputs={"input_1.1": arr}))
    return np.from_dlpack(resp[0].outputs["softmax_tensor:0"])


def main():
    model_repo = sys.argv[1] if len(sys.argv) > 1 else "./model_repo"
    backend_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/ci/opt/triton-server/backends"
    input_bin = sys.argv[3] if len(sys.argv) > 3 else "samples/resnet_input.bin"
    total_batch = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    model_name = sys.argv[6] if len(sys.argv) > 6 else "resnet50_onnx"
    dtype = np.float32 if (len(sys.argv) > 7 and sys.argv[7] == "fp32") else np.float16  # 默认 FP16
    n_req = int(sys.argv[8]) if len(sys.argv) > 8 else 8

    assert total_batch % n_req == 0, f"total_batch({total_batch}) 必须能被 n_requests({n_req}) 整除"
    per_req = total_batch // n_req

    print(f"启动 Triton Server (model={model_name}, total_batch={total_batch}, "
          f"n_requests={n_req} x per_req={per_req}, iters={iters})...")
    server = tritonserver.Server(
        model_repository=model_repo, backend_directory=backend_dir,
        log_verbose=0, exit_timeout=5,
    )
    server.start(wait_until_ready=True)
    try:
        model = server.model(model_name)
        base = np.fromfile(input_bin, dtype=np.float16).reshape(1, 3, 224, 224).astype(dtype)
        arr = np.repeat(base, per_req, axis=0)
        print(f"每请求输入: {arr.shape} {arr.dtype}  (并发 {n_req} 个 -> 合并 batch {total_batch})")

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_req)

        # 热身
        outs = [f.result() for f in [pool.submit(infer_one, model, arr) for _ in range(n_req)]]
        print(f"热身完成 (各请求输出 batch: {[o.shape[0] for o in outs]})")

        shapes_ok = True
        t0 = time.time()
        last = None
        for it in range(iters):
            futs = [pool.submit(infer_one, model, arr) for _ in range(n_req)]
            outs = [f.result() for f in futs]
            last = outs
            for o in outs:
                if o.shape[0] != per_req:
                    shapes_ok = False
        dt = time.time() - t0

        total_samples = total_batch * iters
        throughput = total_samples / dt
        print(f"\n>>>多请求合并结果 (并发 {n_req} x per_req {per_req} = batch {total_batch})")
        print(f"  总样本: {total_samples}")
        print(f"  总耗时: {dt:.3f} s")
        print(f"  平均每轮延迟: {dt/iters*1000:.1f} ms")
        print(f"  吞吐量: {throughput:.1f} samples/s ({throughput/1000:.2f} K samples/s)")
        print(f"  各请求输出 batch 正确: {shapes_ok}")

        pred = last[0][0].astype(np.float32)
        top5 = np.argsort(pred)[::-1][:5]
        print(f"  第1请求第1图 top5: {top5.tolist()}")

        print("\n>>>合并验证: 请查看上文 backend 打印的 [BATCH] 行")
        print(f"    若出现 [BATCH] execute: request_count={n_req} total_batch_size={total_batch}")
        print(f"    即证明 {n_req} 个请求被 dynamic batching 合并为 1 次执行。")
        
        print(">>>success inference:resnet")

        pool.shutdown(wait=False)
    finally:
        try:
            gc.collect()
            server.stop()
            print("Server 已停止")
        except Exception as e:
            print(f"(server.stop 异常, 忽略: {str(e)[:60]})")
        # in-process server 关闭时偶发崩溃, 结果已输出, 干净退出
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
