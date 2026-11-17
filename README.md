# Arcade AI Pet

An offline companion being built for the
[Scratch Arcade OS](https://github.com/ayutaz/scratch-arcade-os).
The goal is simple: speak a short Japanese command, get a reply, and play with
a companion whose behavior has been learned.

[日本語](README.ja.md)

## What works today

The small [Japanese command model](models/kws-v2/model-card.md) now passes a
fresh synthetic evaluation: macro F1 **0.979**, 17,380 bytes of weights.
Its C front end and integer inference match Python decisions on all 956 examples.

The [learned movement policy](models/behavior-v1/model-card.md) also runs in integer C.
On 64 unseen collection boards it averaged 8.64 points versus 0.97 for a seeded random policy.
Its parameters occupy 472 bytes; this is a host benchmark, not handheld performance.

The pet now builds as a **5,142-byte WASM package**. The learned companion,
five voice-result actions and normal exit pass both VM implementations.

![Learned companion](docs/media/companion.gif)

*Actual WASM host replay with simulated voice-result events. Not a device recording.*

The input-offset firmware is installed on one ESP32-S3. Five button actions,
reply playback, player movement and recording cleanup pass automated device checks.
All 16 apps remain intact, and these checks use no Wi-Fi.

**Voice control is not ready.** The corrected firmware still gave no visible
action for five synthetic commands. Two of three environmental recordings also
entered the wait state. These are limited acoustic checks, not human accuracy.
[Device evidence](docs/device-validation.md) records the failures and the input-level
checks; human speech, audible quality and physical display/input validation remain pending.

[Build and preview the pet](docs/pet-app.md).

## Test

Use [uv](https://docs.astral.sh/uv/) with the pinned Python 3.12.13:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

The first suite needs Clang, but no speech engine, GPU, network key or device. Training tests
use a tiny deterministic CPU fixture, not downloaded speech or production weights.

## Run a training experiment

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
[frozen-model recipe](docs/reproduce-model.md), [experiment record](docs/validation.md)
and [data licenses](docs/data-license.md).

Code: [Apache-2.0](LICENSE). Voice models and generated/derived assets have separate
CC BY 3.0 attribution requirements. This repository includes the small trained KWS
weights and derived conformance features. Upstream TTS weights and device
recordings are not included.

Human-voice evaluation: [fixed protocol and anonymous result summary](docs/voice-validation.md).
