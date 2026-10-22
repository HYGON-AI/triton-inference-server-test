#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
test_yolo_grpc.py - YOLOv5s Triton gRPC 端到端网络测试客户端

通过 gRPC 协议 (tritonclient.grpc) 调用远端 Triton Server，
与 test_yolo_http.py (HTTP) 等价，验证 gRPC 服务端到端链路。

用法:
    python3 test_yolo_grpc.py [url] [input.bin]
默认:
    url       = 127.0.0.1:8001
    input.bin = samples/bus_input.bin
"""
import sys
import time

import numpy as np
import tritonclient.grpc as grpcclient

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uni if uni > 0 else 0.0


def nms(boxes, iou_thr=0.45):
    """boxes: [(x1,y1,x2,y2,score,cls)]"""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes
                 if not (b[5] == best[5] and box_iou(best, b) > iou_thr)]
    return keep


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8001"
    input_bin = (sys.argv[2] if len(sys.argv) > 2
                 else "samples/bus_input.bin")

    print(f"gRPC 推理: {url}")

    # 1) 连接 + 健康检查
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_ready():
        print("服务未就绪!")
        sys.exit(1)
    print("服务状态: ready")

    # 2) 模型元信息
    meta = client.get_model_metadata("yolov5s")
    print(f"模型元信息: {meta.name} input={meta.inputs[0].name} "
          f"shape={meta.inputs[0].shape} dtype={meta.inputs[0].datatype}")

    # 3) 读取输入并推理 (binary_data=True 用二进制传输 FP16)
    arr = np.fromfile(input_bin, dtype=np.float16).reshape(1, 3, 640, 640)
    print(f"输入张量: {arr.shape} {arr.dtype} <- {input_bin}")

    inputs = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP16")]
    inputs[0].set_data_from_numpy(arr)
    outputs = [grpcclient.InferRequestedOutput("output0")]

    t0 = time.time()
    result = client.infer("yolov5s", inputs, outputs=outputs)
    dt_ms = (time.time() - t0) * 1000
    out = result.as_numpy("output0")
    print(f"输出张量: {out.shape} {out.dtype}  (gRPC 往返 {dt_ms:.1f} ms)")

    # 4) YOLOv5 后处理: [cx, cy, w, h, obj_conf, 80类] -> 阈值过滤 + NMS
    pred = out[0].astype(np.float32)  # (25200, 85)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    obj = pred[:, 4]
    conf = obj * pred[:, 5:].max(axis=1)
    cls = pred[:, 5:].argmax(axis=1)
    idx = np.where(conf > 0.25)[0]
    boxes = [(cx[i] - w[i] / 2, cy[i] - h[i] / 2,
              cx[i] + w[i] / 2, cy[i] + h[i] / 2,
              float(conf[i]), int(cls[i])) for i in idx]

    print(f"\n>>>gRPC 目标检测推理完成")
    print(f"阈值过滤后候选框数: {len(boxes)}")
    final = nms(boxes, 0.45)
    print(f"NMS 后检测结果数: {len(final)}")
    for k, b in enumerate(final[:10]):
        name = COCO_NAMES[b[5]] if b[5] < len(COCO_NAMES) else "?"
        print(f"  [{k}] class={name}({b[5]}) score={b[4]:.3f} "
              f"box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")
    if len(final)==6:
        print(">>>success inference:{}".format(meta.name))
    else:
        print(">>>fail inference:{}".format(meta.name))


if __name__ == "__main__":
    main()
