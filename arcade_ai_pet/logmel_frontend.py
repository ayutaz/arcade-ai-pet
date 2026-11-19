"""Host-only 64-bin log-mel/energy/duration experiment, without legacy changes."""
from functools import lru_cache
import numpy as np
from .features import SAMPLES, FRAMES, WINDOW, HOP, FFT, _WINDOW, _MEL
from .channel_frontend import active_span

TIME_BINS = 64
MEL_BANDS = 32
FEATURE_COUNT = TIME_BINS * MEL_BANDS + TIME_BINS + 1


@lru_cache(maxsize=FRAMES)
def _weights(frames):
    # Exact integer overlaps: one original frame occupies 64 coordinate units;
    # one output bin occupies `frames` units. Every output bin has positive area.
    bins = np.arange(TIME_BINS)[:, None] * frames
    samples = np.arange(frames)[None, :] * TIME_BINS
    overlap = np.maximum(0, np.minimum(bins + frames, samples + TIME_BINS) - np.maximum(bins, samples))
    return overlap.astype(np.float64) / frames


def pool(values):
    if (not isinstance(values, np.ndarray) or values.ndim != 2 or
        not 1 <= values.shape[0] <= FRAMES or not 1 <= values.shape[1] <= MEL_BANDS or
        values.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(values).all()):
        raise ValueError('finite frame-by-band floating values required')
    return _weights(len(values)) @ values.astype(np.float64)


def features(pcm):
    if not isinstance(pcm, np.ndarray) or pcm.dtype != np.int16 or pcm.shape != (SAMPLES,):
        raise ValueError('exactly 32000 PCM16 samples required')
    x = pcm.astype(np.float64)
    peak = float(abs(x).max())
    if peak < 32:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)
    x /= peak
    framed = np.lib.stride_tricks.sliding_window_view(x, WINDOW)[::HOP]
    energy = np.mean(framed**2, axis=1)
    first, last = active_span(energy)
    duration = 4 * (last - first) / FRAMES
    first, last = max(0, first - 10), min(FRAMES, last + 10)
    power = abs(np.fft.rfft(framed * _WINDOW, n=FFT))**2
    logmel = np.log(np.maximum(power @ _MEL.T, 1e-8))
    bands = pool(logmel[first:last])
    bands = (bands - bands.mean(axis=0)) / 4
    log_energy = np.log(np.maximum(energy, 1e-8))
    log_energy -= log_energy.max()
    energies = pool(log_energy[first:last, None]).ravel() / 2
    return np.clip(np.concatenate((bands.ravel(), energies, [duration])), -16, 16).astype(np.float32)
