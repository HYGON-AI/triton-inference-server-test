# Local dependency fixes

Tested against gRPC v1.81.1 and its Protobuf submodule
`b6f9284da830b69be787732ffdaa35049d20a088` with GCC 11.4.
These patches change source code; they do not disable compiler diagnostics.

- `grpc-state-returns.patch`: six fallback returns after exhaustive enum switches.
  Listed states retain their original strings; future missing cases can still
  produce switch diagnostics because no default cases are added.
- `protobuf-bounded-access.patch`: copy native pointer sizes in array/map getters
  and the tagged-message setter, preserving frozen/presence checks. Compare Java
  field names via a prefix and remaining view instead of removing a suffix.

The inherited CMake toolchain includes `../cmake/dependency-fixes.cmake` after
project initialization. It applies each patch before compilation, handles repeat
configuration, and fails if source changes make the patch incompatible.

Local validation reproduced the original warnings as errors, then compiled the
patched copy/freeze and Java context sources with the same diagnostics enabled.
ASan/UBSan checks cover array/map pointers, hasbits, oneofs, valid/invalid enum
values, and differential string comparisons including binary/empty/random input.
Full integration is validated by the repository's unchanged GitHub CI workflow.
