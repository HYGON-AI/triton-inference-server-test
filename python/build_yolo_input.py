#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
build_yolo_input.py
将任意图片预处理为 YOLOv5s Triton 推理所需的 FP16 输入张量
(1, 3, 640, 640) FP16，NCHW，像素归一化到 [0,1]。

用法:
    python3 build_yolo_input.py <image.jpg> [output.bin]
默认输出: 与图片同名的 .bin（如 bus.jpg -> bus.bin）
"""
import sys
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else img_path.rsplit(".", 1)[0] + ".bin"

    img = Image.open(img_path).convert("RGB")
    img = img.resize((640, 640))                       # 缩放到模型输入尺寸
    arr = np.array(img).astype(np.float32) / 255.0     # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                       # CHW
    arr = np.expand_dims(arr, 0).astype(np.float16)    # (1,3,640,640) FP16
    arr.tofile(out_path)

    print(f"图片 {img_path} -> {out_path}")
    print(f"输出张量: shape={arr.shape} dtype={arr.dtype} bytes={arr.nbytes}")
    print("推理命令: ./test_yolo yolov5s /home/ci/opt/triton-server/backends/ ./model_repo/ <此.bin路径>")


if __name__ == "__main__":
    main()
