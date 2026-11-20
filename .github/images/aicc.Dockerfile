ARG BASE_IMAGE=triton-ci-local:20261110-9ec83ddd
FROM ${BASE_IMAGE}
COPY aicc.run /tmp/ci-aicc/aicc.run
RUN echo '8eb8047d6bc7718d5bbbca6e6f90bff7677f46b659e33a5375d85a444264eb8d  /tmp/ci-aicc/aicc.run' | sha256sum -c - && printf 'y\n' | bash /tmp/ci-aicc/aicc.run --dtk_dir /opt/dtk && /opt/dtk/llvm/bin/clang++ --version && rm -rf /tmp/ci-aicc
ENV CI_HIP_CXX_COMPILER=/opt/dtk/bin/aicc
