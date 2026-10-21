#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""HTTP REST test for the Triton Python backend (python_add model).

Uses only urllib (no tritonclient dependency), same style as test_yolo_http.py.
"""
import json
import sys
import urllib.error
import urllib.request

import numpy as np

MODEL = "python_add"
url = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8000"
base = url if url.startswith("http") else ("http://" + url)


def is_ready(path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def http_post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit("[FAIL] HTTP %s: %s" % (e.code, e.read().decode()))


if not is_ready("/v2/health/ready"):
    raise SystemExit("[FAIL] server not ready")
if not is_ready("/v2/models/%s/ready" % MODEL):
    raise SystemExit("[FAIL] model %s not ready" % MODEL)

# max_batch_size is 0 -> tensor shape is [4]
a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0]

body = {
    "inputs": [
        {"name": "INPUT0", "shape": [4], "datatype": "FP32", "data": a},
        {"name": "INPUT1", "shape": [4], "datatype": "FP32", "data": b},
    ],
    "outputs": [{"name": "OUTPUT0"}],
}

resp = http_post("/v2/models/%s/infer" % MODEL, body)
out = np.array(resp["outputs"][0]["data"], dtype=np.float32)
expect = np.array(a, dtype=np.float32) + np.array(b, dtype=np.float32)

print("INPUT0 :", np.array(a))
print("INPUT1 :", np.array(b))
print("OUTPUT0:", out)
print("EXPECT :", expect)

if not np.allclose(out, expect):
    raise SystemExit("[FAIL] python backend output mismatch")

print("[OK] python backend inference passed")
