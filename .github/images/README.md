# Corrected DTK 26.04 CI compiler

The original SGLang/DTK image's DCC 25.10 (Clang 17) incorrectly diagnoses
valid implicit returns from `void` functions. Use the matching AICC 1.0.0
compiler instead. Do not disable `-Wreturn-type`.

The pinned installer is
`dtk_llvm_aicc-1.0.0+dtk2604.2611111018.g2f732c-x86_64.run`
from the internal resource server. Its SHA-256 is checked by the Dockerfile.
Build on top of the prepared dependency image:

```bash
mkdir -p /tmp/triton-aicc-image
cd /tmp/triton-aicc-image
curl -fSL 'http://42.228.13.241:18000/ai_cc/Nightly/dtk_llvm_aicc-1.0.0%2Bdtk2604.2611111018.g2f732c-x86_64.run' -o aicc.run
docker build -f /path/to/repo/.github/images/aicc.Dockerfile \
  --build-arg BASE_IMAGE=triton-ci-local:20261110-9ec83ddd \
  -t triton-ci-local:20261111-aicc-g2f732c .
```

`CI_HIP_CXX_COMPILER=/opt/dtk/bin/aicc` selects the driver for the server and
all three backends. ARC also selects `/opt/dtk/aillvm/bin/clang++` through
`HIPCXX` for CMake HIP-language kernels and checks both compilation paths.
`build.sh` pairs that frontend with its resource headers in CMake and ABI probes.
Architecture flags remain scoped to the HIP driver;
third-party projects can continue using their own host compiler.

Load the same image onto all eligible runner nodes (or publish it to a
registry accessible to them), then pass this tag as the workflow's
`ci_image` input or set the repository variable `CI_IMAGE` to it.
The build's return-diagnostic preflight accepts valid void returns and
requires both missing and partial non-void returns to be rejected. It uses
`-Werror=return-type` for these probes only, without suppressing diagnostics.
