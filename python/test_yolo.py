#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_yolo.py - 目标检测 (YOLOv5s) Triton Server Python 客户端

通过 tritonserver Python 绑定启动 in-process Triton Server 并进行推理，
与 test_backend_infer.cc 等价，但使用官方 Python API。

用法:
    python3 test_yolo.py [model_repo] [backend_dir] [input.bin]
默认:
    model_repo = ./model_repo
    backend_dir = /home/ci/opt/triton-server/backends
    input.bin = samples/bus_input.bin
"""
import sys
import os
import gc
import numpy as np
from triton_compat import enable_global_symbols
enable_global_symbols()   # 让 libtritonserver 符号全局可见 (替代 LD_PRELOAD)
import tritonserver

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
    model_repo = sys.argv[1] if len(sys.argv) > 1 else "./model_repo"
    backend_dir = (sys.argv[2] if len(sys.argv) > 2
                   else "/home/ci/opt/triton-server/backends")
    input_bin = sys.argv[3] if len(sys.argv) > 3 else "samples/bus_input.bin"

    print(f"启动 Triton Server...\n  model_repo = {model_repo}\n  backend    = {backend_dir}")
    server = tritonserver.Server(
        model_repository=model_repo,
        backend_directory=backend_dir,
        log_verbose=0,
        exit_timeout=5,
    )
    server.start(wait_until_ready=True)
    if 1:
        model = server.model("yolov5s")
        print(f"模型已加载: {model.name}")

        # 加载真实图片预处理后的 FP16 输入 (1,3,640,640)
        arr = np.fromfile(input_bin, dtype=np.float16).reshape(1, 3, 640, 640)
        print(f"输入张量: {arr.shape} {arr.dtype} <- {input_bin}")

        responses = list(model.infer(inputs={"images": arr}))
        out = np.from_dlpack(responses[0].outputs["output0"])
        print(f"输出张量: {out.shape} {out.dtype}\n")

        # YOLOv5 后处理: [cx, cy, w, h, obj_conf, 80类] -> 阈值过滤 + NMS
        pred = out[0].astype(np.float32)  # (25200, 85)
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        obj = pred[:, 4]
        conf = obj * pred[:, 5:].max(axis=1)
        cls = pred[:, 5:].argmax(axis=1)
        idx = np.where(conf > 0.25)[0]
        boxes = [(cx[i] - w[i] / 2, cy[i] - h[i] / 2,
                  cx[i] + w[i] / 2, cy[i] + h[i] / 2,
                  float(conf[i]), int(cls[i])) for i in idx]

        print(f">>>目标检测推理完成")
        print(f"阈值过滤后候选框数: {len(boxes)}")
        final = nms(boxes, 0.45)
        print(f"NMS 后检测结果数: {len(final)}")
        for k, b in enumerate(final[:10]):
            name = COCO_NAMES[b[5]] if b[5] < len(COCO_NAMES) else "?"
            print(f"  [{k}] class={name}({b[5]}) score={b[4]:.3f} "
                  f"box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")
        if len(final)==6:
           print(">>>success inference:{}".format(model.name))
        else:
           print(">>>fail inference:{}".format(model.name))

    if 1:
        # 关键：释放推理响应（np.from_dlpack 的数组持有 response 缓冲），
        # 否则模型卸载时 response 无法释放 -> stop() 挂起
        try:
            del out, responses
        except Exception:
            pass
        gc.collect()
        server.stop()
        print("Server 已停止")


if __name__ == "__main__":
    main()
