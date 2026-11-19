"""Development PoC: utterance-centered aggregated MFCCs, energy/duration intact."""
import numpy as np
from .duration_frontend import features as original


def center(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or
        values.ndim not in (1, 2) or values.shape[-1] != 261 or not values.size or
        not np.isfinite(values).all() or np.any(abs(values) > 16) or
        np.any((values[..., -1] < 0) | (values[..., -1] > 4))):
        raise ValueError('finite bounded float32 context-duration vectors required')
    result = values.copy()
    mfcc = values[..., :240].reshape(values.shape[:-1] + (20, 12)).astype(np.float64)
    centered = np.clip(mfcc - mfcc.mean(axis=-2, keepdims=True), -16, 16)
    result[..., :240] = centered.reshape(values.shape[:-1] + (240,)).astype(np.float32)
    return result


def features(pcm):
    return center(original(pcm, context=True))
