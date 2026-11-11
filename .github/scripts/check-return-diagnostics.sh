#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

compiler="${CI_HIP_CXX_COMPILER:-/opt/dtk/bin/hipcc}"
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
flags=(-Wall -Wextra -Werror=return-type -c)
if ! "$compiler" "${flags[@]}" "$probe_dir/valid.cc" -o "$probe_dir/valid.o" > "$probe_dir/valid.log" 2>&1; then
    cat "$probe_dir/valid.log" >&2
    echo '::error::Compiler rejects valid void returns; use a corrected HCU compiler.' >&2
    exit 1
fi
for source in missing partial; do
    if "$compiler" "${flags[@]}" "$probe_dir/$source.cc" -o "$probe_dir/$source.o" > "$probe_dir/$source.log" 2>&1; then
        echo "::error::Compiler failed to reject $source return value." >&2
        exit 1
    fi
    if ! grep -Fq -- '-Wreturn-type' "$probe_dir/$source.log"; then
        cat "$probe_dir/$source.log" >&2
        echo '::error::Invalid return probe failed for an unexpected reason.' >&2
        exit 1
    fi
done
printf 'Return diagnostics verified: %s (valid void accepted; missing and partial returns rejected).\n' "$compiler"
