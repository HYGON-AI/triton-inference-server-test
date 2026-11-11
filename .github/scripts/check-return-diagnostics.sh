#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

probe_dir="$(mktemp -d)"
trap 'rm -rf -- "$probe_dir"' EXIT
cat > "$probe_dir/valid.cc" <<'CPP'
void implicit_return() {}
void explicit_return() { return; }
struct Example { void method() {} };
int valid_return() { return 1; }
CPP
cat > "$probe_dir/missing.cc" <<'CPP'
int missing_return() {}
CPP
cat > "$probe_dir/partial.cc" <<'CPP'
int partial_return(int value) { if (value) return 1; }
CPP

# Test the same driver and architecture flags as the build. Keep return-type
# diagnostics enabled; a blanket suppression would also hide actual defects.
check_driver() {
    local compiler="$1"
    shift
    local flags=("$@" -Wall -Wextra -Werror=return-type -c)
    if ! "$compiler" "${flags[@]}" "$probe_dir/valid.cc" -o "$probe_dir/valid.o" > "$probe_dir/valid.log" 2>&1; then
        cat "$probe_dir/valid.log" >&2
        echo '::error::Compiler rejects valid void returns; use a corrected HCU compiler.' >&2
        return 1
    fi
    local source
    for source in missing partial; do
        if "$compiler" "${flags[@]}" "$probe_dir/$source.cc" -o "$probe_dir/$source.o" > "$probe_dir/$source.log" 2>&1; then
            echo "::error::Compiler failed to reject $source return value." >&2
            return 1
        fi
        if ! grep -Fq -- '-Wreturn-type' "$probe_dir/$source.log"; then
            cat "$probe_dir/$source.log" >&2
            echo '::error::Invalid return probe failed for an unexpected reason.' >&2
            return 1
        fi
    done
    printf 'Return diagnostics verified: %s (valid void accepted; missing and partial returns rejected).\n' "$compiler"
}
check_driver "${CI_HIP_CXX_COMPILER:-/opt/dtk/bin/hipcc}"
if [[ -n "${HIPCXX:-}" ]]; then
    target="${CI_HIP_ARCHITECTURES:-${AMDGPU_TARGETS:-gfx938}}"
    target="${target%%[;, ]*}"
    check_driver "$HIPCXX" -x hip --rocm-path=/opt/dtk "--offload-arch=$target"
fi
