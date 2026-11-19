"""Host-only fixed temporal CNN: int8 parameters and Q10 activations."""
import numpy as np

SHAPES = ((24,65), (32,72), (32,96), (64,160), (7,64))
KERNELS = (5,3,3)


def sequence(x):
    if (not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 260 or not len(x) or
            not np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.complexfloating) or
            not np.isfinite(x).all() or np.any((x < -16) | (x > 16))):
        raise ValueError('finite bounded 260-value feature rows required')
    return np.concatenate((x[:,:240].reshape(-1,20,12), x[:,240:,None]), axis=2)


def weights_bytes(layers):
    if type(layers) is not list or len(layers) != 5:
        raise ValueError('fixed five-layer temporal model required')
    for layer, shape in zip(layers, SHAPES, strict=True):
        if (type(layer) is not dict or set(layer) != {'weight','bias','shift'} or
                not isinstance(layer['weight'], np.ndarray) or layer['weight'].dtype != np.int8 or layer['weight'].shape != shape or
                not isinstance(layer['bias'], np.ndarray) or layer['bias'].dtype != np.int32 or layer['bias'].shape != (shape[0],) or
                isinstance(layer['shift'], (bool,np.bool_)) or not isinstance(layer['shift'], (int,np.integer)) or not 0 <= layer['shift'] <= 16):
            raise ValueError('bounded int8/int32 temporal layer required')
    size = sum(layer['weight'].nbytes + layer['bias'].nbytes + 4 for layer in layers)
    if size > 24576:
        raise ValueError('24KiB numeric weight budget exceeded')
    return size


def _apply(q, layer, *, relu):
    result = (q @ layer['weight'].astype(np.int64).T + layer['bias']) >> int(layer['shift'])
    if relu:
        return np.clip(result, 0, 6144)
    if np.any((result < -(2**31)) | (result > 2**31-1)):
        raise ValueError('integer logit overflow')
    return result


def scores(x, layers):
    weights_bytes(layers)
    q = np.rint(sequence(x)*1024).astype(np.int64)
    for layer, kernel in zip(layers[:3], KERNELS, strict=True):
        padded = np.pad(q, ((0,0), (kernel//2,kernel//2), (0,0)))
        # NumPy layout is batch/time/input-channel/kernel, matching Torch's weight order.
        patches = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=1)
        q = _apply(patches.reshape(len(q),20,-1), layer, relu=True)
    q = (q.reshape(len(q),5,4,32).sum(axis=2) >> 2).reshape(len(q),160)
    q = _apply(q, layers[3], relu=True)
    return _apply(q, layers[4], relu=False)
