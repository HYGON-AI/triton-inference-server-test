#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Validate the supplied HCU image and share runtime metadata between ARC jobs."""

import importlib.metadata as metadata
import json
import platform
import re
import sys
from pathlib import Path

FRAMEWORKS = ("torch", "onnxruntime", "numpy")
BINDINGS = {"tritonserver", "tritonfrontend"}


def image_metadata():
    versions = {name: metadata.version(name) for name in FRAMEWORKS}
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("CI_IMAGE must provide Python 3.12")
    if versions["torch"].split("+", 1)[0] != "2.7.1":
        raise RuntimeError("Current ONNX Runtime requires HCU Torch 2.7.1")
    if versions["onnxruntime"].split("+", 1)[0] != "1.19.2":
        raise RuntimeError("CI_IMAGE must provide the matching HCU ONNX Runtime 1.19.2")
    if versions["numpy"].split(".", 1)[0] != "1":
        raise RuntimeError("Current HCU ONNX Runtime environment requires numpy<2")
    import onnxruntime
    import torch

    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError("CI_IMAGE must provide HIP/HCU Torch, not a CUDA/CPU build")
    if not Path("/opt/dtk/lib/libmigraphx_c.so.3").exists():
        raise RuntimeError("CI_IMAGE must provide the matching MIGraphX runtime")
    return {
        "python": list(sys.version_info[:2]),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "versions": versions,
        "hip": torch.version.hip,
        "onnx_providers": onnxruntime.get_available_providers(),
    }


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    command = sys.argv[1]
    directory = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if command == "requirements":
        # Avoid pip freeze's build-pod-local file:// URLs.
        requirements = {}
        for distribution in metadata.distributions():
            name = distribution.metadata["Name"]
            if name and normalize_name(name) not in BINDINGS:
                requirements[normalize_name(name)] = f"{name}=={distribution.version}"
        (directory / "python-requirements.txt").write_text(
            "\n".join(requirements[name] for name in sorted(requirements)) + "\n"
        )
        return
    actual = image_metadata()
    if command == "check-image":
        print(json.dumps(actual, indent=2))
    elif command == "snapshot":
        (directory / "image-runtime.json").write_text(json.dumps(actual, indent=2) + "\n")
        (directory / "framework-constraints.txt").write_text(
            "".join(f"{name}=={actual['versions'][name]}\n" for name in FRAMEWORKS)
        )
    elif command == "verify-image":
        expected = json.loads((directory / "image-runtime.json").read_text())
        if actual != expected:
            raise RuntimeError(f"Build/test runtime mismatch: expected {expected}, got {actual}")
    elif command == "check-device":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No usable HCU is visible in the single-card test pod")
        print(f"Visible HCU devices: {torch.cuda.device_count()}; tests use logical device 0")
    else:
        raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    try:
        main()
    except (metadata.PackageNotFoundError, ImportError, RuntimeError) as error:
        raise SystemExit(
            f"::error::{error}. Configure CI_IMAGE with matching HCU build/runtime dependencies."
        ) from error
