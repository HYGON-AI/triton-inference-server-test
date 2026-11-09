#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORK_BASE
WORK_BASE="$(dirname -- "$repo_root")"
shared_root="/ci_public/${GITHUB_REPOSITORY#*/}/runs"

build_dir() {
    printf '%s/%s-%s\n' "$shared_root" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
}

test_dir() {
    local dir="${CI_ARTIFACT_DIR:?build job did not publish an artifact directory}"
    if [[ "$dir" != "$shared_root/$GITHUB_RUN_ID-"* || ! -f "$dir/complete" ]]; then
        echo '::error::Published build is missing or belongs to another run.' >&2
        return 1
    fi
    printf '%s\n' "$dir"
}

prepare_image() {
    [[ -f /opt/dtk/env.sh ]] || { echo '::error::CI_IMAGE must contain DTK.' >&2; return 1; }
    set +u
    # shellcheck disable=SC1091
    source /opt/dtk/env.sh
    # shellcheck source=env.sh
    source "$repo_root/env.sh"
    set -u
    python3 "$repo_root/.github/scripts/arc_runtime.py" check-image
}

case "${1:-}" in
    build)
        [[ -d /ci_public ]] || { echo '::error::ARC must mount shared storage at /ci_public.' >&2; exit 1; }
        run_dir="$(build_dir)"
        mkdir -p "$run_dir"
        echo "artifact_dir=$run_dir" >> "$GITHUB_OUTPUT"
        prepare_image
        if [[ -n "${CI_REPO_TOKEN:-}" ]]; then
            git config --global credential.https://github.com.helper \
                '!f() { if [ "$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$CI_REPO_TOKEN"; fi; }; f'
        fi
        if [[ -n "${CI_HIP_ARCHITECTURES:-}" ]]; then
            export AMDGPU_TARGETS="$CI_HIP_ARCHITECTURES"
        fi
        if [[ -n "${AMDGPU_TARGETS:-}" ]]; then
            targets="${AMDGPU_TARGETS//;/ }"
            read -r -a architectures <<< "${targets//,/ }"
            for architecture in "${architectures[@]}"; do
                # All backend CMake projects use hipcc as their C++ compiler.
                export CXXFLAGS="${CXXFLAGS:-} --offload-arch=$architecture"
            done
        fi
        echo "HIP compilation targets: ${AMDGPU_TARGETS:-image/compiler default}"
        export CI_SKIP_DEPS=1 CI_UPDATE_DEPS=1
        # An identical absolute prefix on both pods preserves native RPATHs.
        export CI_INSTALL_PREFIX="$run_dir/install"
        export LD_LIBRARY_PATH="$CI_INSTALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
        python3 "$repo_root/.github/scripts/arc_runtime.py" snapshot "$run_dir"
        export PIP_CONSTRAINT="$run_dir/framework-constraints.txt"
        bash -e "$repo_root/build.sh" "$WORK_BASE" HYGON-AI 2>&1 | tee "$run_dir/build.log"
        python3 "$repo_root/.github/scripts/arc_runtime.py" verify-image "$run_dir"
        python3 "$repo_root/.github/scripts/arc_runtime.py" requirements "$run_dir"
        for required in bin/tritonserver backends/onnx/libtriton_onnx.so backends/migraphx/libtriton_migraphx.so backends/pytorch/libtriton_pytorch.so; do
            [[ -f "$CI_INSTALL_PREFIX/$required" ]] || { echo "::error::Missing install artifact: $required" >&2; exit 1; }
        done
        for binary in "$CI_INSTALL_PREFIX/bin/tritonserver" "$CI_INSTALL_PREFIX"/backends/*/*.so; do
            [[ -f "$binary" ]] || continue
            dependencies="$(ldd "$binary")"
            if grep -Fq 'not found' <<< "$dependencies"; then
                printf '%s\n%s\n' "$binary" "$dependencies" >&2
                exit 1
            fi
        done
        printf '%s\n' "$GITHUB_SHA" > "$run_dir/test-commit"
        find "$CI_INSTALL_PREFIX" -type f -print0 | sort -z | xargs -0 sha256sum > "$run_dir/install.sha256"
        touch "$run_dir/complete"
        ;;
    test)
        run_dir="$(test_dir)"
        [[ "$(cat "$run_dir/test-commit")" = "$GITHUB_SHA" ]] || { echo '::error::Build commit mismatch.' >&2; exit 1; }
        sha256sum --check --quiet "$run_dir/install.sha256"
        prepare_image
        export LD_LIBRARY_PATH="$run_dir/install/lib:${LD_LIBRARY_PATH:-}"
        python3 "$repo_root/.github/scripts/arc_runtime.py" verify-image "$run_dir"
        export PIP_CONSTRAINT="$run_dir/framework-constraints.txt"
        python3 -m pip install -r "$run_dir/python-requirements.txt"
        shopt -s nullglob
        wheels=("$run_dir/install/python/"*.whl)
        (( ${#wheels[@]} > 0 )) || { echo '::error::Published Python bindings are missing.' >&2; exit 1; }
        python3 -m pip install --no-deps --force-reinstall "${wheels[@]}"
        python3 "$repo_root/.github/scripts/arc_runtime.py" verify-image "$run_dir"
        python3 "$repo_root/.github/scripts/arc_runtime.py" check-device
        mkdir -p "$WORK_BASE/opt"
        install_link="$WORK_BASE/opt/triton-server"
        if [[ -e "$install_link" || -L "$install_link" ]]; then
            [[ "$(realpath -- "$install_link")" = "$run_dir/install" ]] || { echo '::error::Unexpected existing install directory.' >&2; exit 1; }
        else
            ln -s "$run_dir/install" "$install_link"
        fi
        bash -e "$repo_root/run_test.sh" "$WORK_BASE" 2>&1 | tee "$run_dir/test.log"
        ;;
    save-build-log)
        run_dir="$(build_dir)"
        echo "Build artifacts and logs: $run_dir"
        ;;
    save-test-logs)
        run_dir="$(test_dir)"
        for log in /tmp/triton_http.log /tmp/triton_pytorch.log; do
            [[ ! -f "$log" ]] || cp -- "$log" "$run_dir/"
        done
        echo "Test artifacts and logs: $run_dir"
        ;;
    *)
        echo 'Usage: arc-ci.sh {build|test|save-build-log|save-test-logs}' >&2
        exit 2
        ;;
esac
