# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""triton_compat.py - 让 in-process tritonserver 正确加载 backend 共享库

背景:
  Python 默认以 RTLD_LOCAL 加载 C 扩展 (triton_bindings.so), 其依赖的
  libtritonserver.so 的 TRITONBACKEND_* 符号不会进入全局符号表。之后
  Triton 在进程内 dlopen backend (libtriton_*.so) 时, backend 里对这些
  符号的引用无法解析, 报错:
      unable to load shared library: .../libtriton_xxx.so:
      undefined symbol: TRITONBACKEND_ModelInstanceModel

用法:
  在 `import tritonserver` 之前调用 enable_global_symbols():
      from triton_compat import enable_global_symbols
      enable_global_symbols()
      import tritonserver

  用于替代 LD_PRELOAD=libtritonserver.so。
"""
import os
import sys


def enable_global_symbols():
    """让后续 dlopen 的库 (及其依赖) 符号全局可见 (RTLD_GLOBAL)。"""
    flags = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    try:
        sys.setdlopenflags(sys.getdlopenflags() | flags)
    except (AttributeError, OSError):
        pass
