# Host teacher environment

This optional uv project locks the dependencies used for local Irodori v2 VoiceDesign generation and independent Whisper content checks. It is separate from the device model's core Python environment. No model weights or audio are included.

The tested host is macOS/Apple Silicon with Python 3.12.13, CPU/FP32 Irodori and CPU/int8 Whisper. Other platforms are not validated by this lockfile alone. This environment does not run on the ESP32-S3.

## Setup

From the repository root:

```sh
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked --project tools/neural-teacher
uv lock --check
uv lock --check --project tools/neural-teacher
```

The tested Mac also needed CMake, pkg-config and GNU coreutils (`nproc`) for native dependencies. Use the explicit workspace boundary in this project's pyproject; do not add it to the core workspace or replace the core NumPy pin.

Obtain Irodori's source separately, at commit `2708d3cadf726d4389d25eb4bb7a0344517a9a40` from [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS/tree/2708d3cadf726d4389d25eb4bb7a0344517a9a40). The locked `dacvae` dependency uses commit `414c20785fc3a28373073ea8ef7a1316eeeaca6e`. No Irodori implementation is copied into this repository.

Use the fixed model-card revisions in [provider provenance](../../docs/neural-provider-model-cards.json) and verify the exact model/codec/ASR file hashes declared by the generation plan. The teacher loads about 2.47 GB of model and codec weights; the separate Whisper model is about 484 MB. Obtain and review these assets before inference, then use local-only paths and an offline Hugging Face cache. No device recordings are required.

## Execution boundary

- Provider environment: explicitly load `InferenceRuntime`, `SamplingRequest` and `WhisperModel`, then inject them into `host_teacher.source_callbacks`. Run `generate_development(..., finalize=False)`.
- Core environment: run `finalize_development(output)` with the repository's normal `uv run --locked python`. Its NumPy pin supplies the feature-generation version.
- A final-condition run requires a verified frozen model; see the final-corpus API when available. Never use final conditions to select a checkpoint or threshold.

See [generation contracts](../../docs/neural-corpus.md) and [usage conditions](../../docs/data-license.md). The environment lock makes dependency resolution reproducible; it does not prove generated content, independent speaker identity, or real-device recognition quality.

## Verification

2026-11-19: copied the actual isolated provider pyproject/lock without changing dependency versions. Both nested and core `uv lock --check` succeeded. The same provider lock is executing a CPU development-generation run; source generation is an experiment, not a completed quality result. No new product behavior is implemented here, so dependency synchronization and identity checks are validation rather than a claimed TDD Red/Green cycle.
