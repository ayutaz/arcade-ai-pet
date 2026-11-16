# Arcade AI Pet

An offline companion being built for the
[Scratch Arcade OS](https://github.com/ayutaz/scratch-arcade-os).
The goal is simple: speak a short Japanese command, get a reply, and play with
a companion whose behavior has been learned.

[日本語](README.ja.md)

## What works today

The host research code generates licensed synthetic speech, extracts small
audio features, trains a compact classifier, and converts its dense layers to
integer inference. It recognizes five command classes plus unknown and silence.

**The pet, voice service and on-device recognition are not finished.** The first
model failed the acceptance target. An alignment change passes development
validation, but still needs a fresh held-out evaluation and real speech tests.
No firmware has been installed by this project.

## Test

Use [uv](https://docs.astral.sh/uv/) with the pinned Python 3.12.13:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

The first suite needs no speech engine, GPU, network key or device. Training tests
use a tiny deterministic CPU fixture, not downloaded speech or production weights.

## Reproduce the first experiment

On macOS, install Open JTalk with Homebrew. Keep generated speech and models local:

```sh
brew install open-jtalk
uv run --locked --group train python -m arcade_ai_pet.data \
  --jtalk-root /opt/homebrew/opt/open-jtalk --output datasets/my-speech
uv run --locked --group train python -m arcade_ai_pet.training \
  --dataset datasets/my-speech --output checkpoints/my-model
```

The recipe uses a 260-value aligned MFCC/energy front end and a small MLP.
Each source utterance belongs to one split before augmentation. The checkpoint
and rejection threshold are selected using validation only. Current synthetic
splits share the same voice identities; they do not measure unseen human speakers.

This runs on the computer's CPU and makes no Vast.ai API call. Cloud training is
not enabled. Input audio is limited to two seconds of mono PCM16 at 16 kHz.
Read the [acceptance targets](docs/acceptance.md),
[experiment record](docs/validation.md) and [data licenses](docs/data-license.md).

Code: [Apache-2.0](LICENSE). Voice models and generated/derived assets have separate
CC BY 3.0 attribution requirements. No voice weights or recorded device audio are
included in this repository.
