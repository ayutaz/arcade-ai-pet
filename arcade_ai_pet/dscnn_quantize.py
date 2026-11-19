"""Bounded channel-wise power-of-two quantization for the fixed DS-CNN."""
import math
import numpy as np


def _float(values):
    if (not isinstance(values, np.ndarray) or values.dtype not in (np.dtype('float32'), np.dtype('float64')) or
        not values.size or not np.isfinite(values).all() or np.any(abs(values) > 1e6)):
        raise ValueError('finite bounded floating parameters required')
    return values.astype(np.float64)


def _parameters(weight, bias):
    weight, bias = _float(weight), _float(bias)
    if (weight.ndim not in (2,4) or not 1 <= weight.shape[0] <= 64 or
        weight.size > 24000 or bias.shape != (weight.shape[0],)):
        raise ValueError('bounded output-channel parameter shapes required')
    return weight, bias


def quantize_layer(weight, bias):
    weight, bias = _parameters(weight, bias)
    maximum = abs(weight.reshape(len(bias), -1)).max(axis=1)
    shift = np.clip(np.floor(np.log2(127 / np.maximum(maximum, 1e-12))), 0, 16).astype(np.int32)
    scale = np.exp2(shift).astype(np.float64)
    shaped = scale.reshape((len(bias),) + (1,) * (weight.ndim - 1))
    return dict(weight=np.clip(np.rint(weight * shaped), -127, 127).astype(np.int8),
                bias=np.clip(np.rint(bias * 32 * scale), -(2**31), 2**31-1).astype(np.int32),
                shift=shift)


def fold_batch_norm(weight, bias, mean, variance, gamma, beta, *, eps):
    weight, bias = _parameters(weight, bias)
    vectors = [_float(v) for v in (mean, variance, gamma, beta)]
    if (any(v.shape != bias.shape for v in vectors) or (vectors[1] < 0).any() or
        type(eps) not in (int, float) or not math.isfinite(eps) or not 1e-12 <= eps <= 1):
        raise ValueError('matching BN vectors, nonnegative variance and positive epsilon required')
    mean, variance, gamma, beta = vectors
    scale = gamma / np.sqrt(variance + eps)
    return weight * scale.reshape((len(bias),) + (1,) * (weight.ndim - 1)), beta + (bias - mean) * scale
