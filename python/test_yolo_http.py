#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_yolo_http.py - YOLOv5s Triton HTTP 端到端网络测试客户端

通过 HTTP 协议 (POST /v2/models/yolov5s/infer) 调用远端 Triton Server，
与 test_yolo.py (in-process C API) 等价，验证 HTTP 服务端到端链路。

用法:
    python3 test_yolo_http.py [url] [input.bin]
默认:
    url       = http://127.0.0.1:8000
    input.bin = samples/bus_input.bin
"""
import sys
import json
import struct
import urllib.request

import numpy as np

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


def _triton_datatype(dtype):
    """numpy dtype -> Triton datatype 字符串"""
    return {
        np.float16: "FP16", np.float32: "FP32", np.float64: "FP64",
        np.int8: "INT8", np.int16: "INT16", np.int32: "INT32",
        np.int64: "INT64", np.uint8: "UINT8", np.uint16: "UINT16",
        np.uint32: "UINT32", np.uint64: "UINT64",
        np.bool_: "BOOL",
    }.get(np.dtype(dtype).type, np.dtype(dtype).name.upper())


def http_infer(url, model, inputs, output_names):
    """发送 Triton HTTP binary 请求，返回输出 dict {name: np.ndarray}"""
    # 构造 JSON 头（Triton 3.x binary 协议：JSON 头 + 二进制，无分隔符）
    in_desc = []
    body = b""
    for name, arr in inputs.items():
        binary = np.ascontiguousarray(arr).tobytes()
        in_desc.append({
            "name": name,
            "shape": list(arr.shape),
            "datatype": _triton_datatype(arr.dtype),
            "parameters": {"binary_data_size": len(binary)},
        })
        body += binary
    request_json = json.dumps({
        "inputs": in_desc,
        "outputs": [
            {"name": n, "parameters": {"binary_data": True}}
            for n in output_names
        ],
    }).encode("utf-8")
    # Triton 3.x：body = JSON 头 + 二进制（无分隔符），
    # JSON 头长度通过 Inference-Header-Content-Length HTTP 头指定
    body = request_json + body

    req = urllib.request.Request(
        f"{url}/v2/models/{model}/infer",
        data=body,
        headers={
            "Content-Type": "application/octet-stream",
            # Triton 3.x: JSON 头长度通过 HTTP 头指定（不再是 2.x 的 \n 分隔）
            "Inference-Header-Content-Length": str(len(request_json)),
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        # Triton 3.x: 响应也用 Inference-Header-Content-Length 头指定 JSON 头长度
        header_len = int(resp.headers.get("Inference-Header-Content-Length", "0"))
        header = json.loads(raw[:header_len].decode("utf-8"))
        bin_data = raw[header_len:]
    outputs = {}
    for out in header["outputs"]:
        size = out["parameters"]["binary_data_size"]
        dt = _numpy_dtype(out["datatype"])
        nbytes = int(np.prod(out["shape"])) * dt.itemsize
        data = np.frombuffer(bin_data[:nbytes], dtype=dt).reshape(out["shape"])
        bin_data = bin_data[nbytes:]
        outputs[out["name"]] = data
    return outputs


def _numpy_dtype(triton_type):
    """Triton datatype 字符串 -> numpy dtype"""
    mapping = {
        "FP16": np.float16, "FP32": np.float32, "FP64": np.float64,
        "INT8": np.int8, "INT16": np.int16, "INT32": np.int32,
        "INT64": np.int64, "UINT8": np.uint8, "UINT16": np.uint16,
        "UINT32": np.uint32, "UINT64": np.uint64, "BOOL": np.bool_,
    }
    key = triton_type.upper()
    if key in mapping:
        return np.dtype(mapping[key])
    return np.dtype(triton_type.lower())


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    input_bin = (sys.argv[2] if len(sys.argv) > 2
                 else "samples/bus_input.bin")

    print(f"HTTP 推理: {url}/v2/models/yolov5s/infer")

    # 1) 健康检查
    try:
        with urllib.request.urlopen(f"{url}/v2/health/ready", timeout=10) as r:
            print(f"服务状态: ready={r.read().decode()} ({r.status})")
    except Exception as e:
        print(f"健康检查失败: {e}")
        sys.exit(1)

    # 2) 模型元信息
    try:
        with urllib.request.urlopen(f"{url}/v2/models/yolov5s", timeout=10) as r:
            meta = json.loads(r.read().decode())
            print(f"模型元信息: {json.dumps(meta, indent=2)}")
    except Exception as e:
        print(f"模型信息获取失败: {e}")

    # 3) 读取输入并推理
    arr = np.fromfile(input_bin, dtype=np.float16).reshape(1, 3, 640, 640)
    print(f"输入张量: {arr.shape} {arr.dtype} <- {input_bin}")

    import time
    t0 = time.time()
    outputs = http_infer(url, "yolov5s", {"images": arr}, ["output0"])
    dt_ms = (time.time() - t0) * 1000
    out = outputs["output0"]
    print(f"输出张量: {out.shape} {out.dtype}  (HTTP 往返 {dt_ms:.1f} ms)")

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

    print(f"\n>>>HTTP 目标检测推理完成")
    print(f"阈值过滤后候选框数: {len(boxes)}")
    final = nms(boxes, 0.45)
    print(f"NMS 后检测结果数: {len(final)}")
    for k, b in enumerate(final[:10]):
        name = COCO_NAMES[b[5]] if b[5] < len(COCO_NAMES) else "?"
        print(f"  [{k}] class={name}({b[5]}) score={b[4]:.3f} "
              f"box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")

    if len(final)==6:
        print(">>>success inference:{}".format(meta["name"]))
    else:
        print(">>>fail inference:{}".format(meta["name"]))



if __name__ == "__main__":
    main()
