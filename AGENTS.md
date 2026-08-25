# Agent Guide

This is Kohan's owned fork of `pwilkin/trellis.cpp`. `origin` is the writable
fork and `upstream` is read-only for synchronization. The matching upstream
reference checkout is under `src/references/game-dev/trellis/trellis.cpp`.

Keep changes narrow, portable across Linux and Windows, and compatible with
CPU, Vulkan, ROCm, and CUDA builds. Do not modify model math, dtype policy, GPU
dispatch, or checkpoint-owned tensors for a CLI/integration feature.

For resumable generation:

- Checkpoints must be explicit, versioned, bounds-checked, and written
  atomically through a temporary file in the destination directory.
- A checkpoint must contain enough metadata and state to produce deterministic
  continuation without silently combining incompatible stages or settings.
- Invalid, truncated, unsupported, or mismatched checkpoints must fail clearly.
- Preserve existing `--dump-slat` debug behavior and existing CLI invocation.
- Do not call diagnostic dumps resumable unless the production CLI can load
  them and complete the remaining requested pipeline.
- Keep large checkpoint files out of Git.

Build in a separate build directory. Start with parser and checkpoint unit tests;
do not run a full multi-minute model generation unless explicitly requested.
