#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

image="${CI_IMAGE:?CI_IMAGE is required}"
archive_dir="${CI_IMAGE_ARCHIVE_DIR:-/ci_public/ci-images/triton-inference-server-test-ci/20261111-aicc-g2f732c}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
credential_file=''
cleanup() { [[ -z "$credential_file" ]] || rm -f -- "$credential_file"; }
trap cleanup EXIT
if [[ -n "${CI_K8S_READONLY_KUBECONFIG:-}" ]]; then
    credential_file=$(umask 077; mktemp "${RUNNER_TEMP:-/tmp}/triton-kubeconfig.XXXXXX")
    printf '%s\n' "$CI_K8S_READONLY_KUBECONFIG" > "$credential_file"
    unset CI_K8S_READONLY_KUBECONFIG
    export KUBECONFIG="$credential_file"
elif [[ -n "${CI_KUBECONFIG_PATH:-}" ]]; then
    export KUBECONFIG="$CI_KUBECONFIG_PATH"
fi
targets=$(python3 "$script_dir/discover-ci-image-nodes.py")
[[ -n "$targets" ]] || { echo 'Dynamic discovery returned no nodes.' >&2; exit 1; }
mapfile -t nodes <<< "$targets"
[[ -d "$archive_dir" ]] || { echo "Shared image directory not found: $archive_dir" >&2; exit 1; }
archive_image=$(cat "$archive_dir/image-tag")
expected_id=$(cat "$archive_dir/image-id")
read -r archive_sha _ < "$archive_dir/image.tar.gz.sha256"
[[ "$image" == "$archive_image" ]] || {
    echo "Requested image $image does not match shared archive $archive_image. Configure CI_IMAGE_ARCHIVE_DIR for this image." >&2
    exit 1
}
[[ "$expected_id" =~ ^sha256:[a-f0-9]{64}$ && "$archive_sha" =~ ^[a-f0-9]{64}$ ]] || {
    echo 'Invalid shared image metadata.' >&2
    exit 1
}

for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[a-zA-Z0-9][a-zA-Z0-9.:-]*$ ]] || { echo "Invalid node: $node" >&2; exit 1; }
    printf '\nPreparing CI image on %s\n' "$node"
    printf -v remote_command 'bash -s -- %q %q %q %q' "$image" "$archive_dir" "$expected_id" "$archive_sha"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 \
        "github@$node" "$remote_command" <<'REMOTE'
set -euo pipefail
image=$1
archive_dir=$2
expected_id=$3
archive_sha=$4
export PATH=/usr/local/bin:/usr/bin:/bin
docker info >/dev/null
state_dir="$HOME/.cache/triton-inference-server-test-ci"
mkdir -p "$state_dir"
exec 9>"$state_dir/image-load.lock"
flock -w 1500 9
actual=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)
if [[ "$actual" == "$expected_id" ]]; then
    echo "$(hostname): matching image is present; skipping load."
    exit 0
fi
if [[ -n "$actual" ]]; then
    echo "$(hostname): refusing to overwrite $image (expected $expected_id, found $actual)." >&2
    exit 1
fi
echo "$(hostname): image is missing; verifying and loading shared archive."
cd "$archive_dir"
printf '%s  image.tar.gz\n' "$archive_sha" | sha256sum --check --strict
docker image load --input image.tar.gz
actual=$(docker image inspect --format '{{.Id}}' "$image")
[[ "$actual" == "$expected_id" ]] || { echo "Loaded image ID mismatch: $actual" >&2; exit 1; }
echo "$(hostname): shared image loaded and verified."
REMOTE
done
