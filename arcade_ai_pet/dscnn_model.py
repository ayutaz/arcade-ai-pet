"""Fixed six-layer integer DS-CNN host reference. No C export or device access."""
import numpy as np

FEATURE_COUNT = 2113
SHAPES = ((8,1,3,3), (8,1,3,3), (16,8,1,1), (16,1,3,3), (24,16,1,1), (7,3137))
STRIDES = (2,1,1,2,1)
GROUPS = (1,8,1,16,1)
BATCH = 32


def validate(layers):
    if not isinstance(layers, (list, tuple)) or len(layers) != 6:
        raise ValueError('exactly six fixed CNN layers required')
    for row, shape in zip(layers, SHAPES, strict=True):
        if (type(row) is not dict or set(row) != {'weight', 'bias', 'shift'} or
            any(not isinstance(row[k], np.ndarray) for k in row) or
            row['weight'].dtype != np.int8 or row['weight'].shape != shape or
            row['bias'].dtype != np.int32 or row['bias'].shape != (shape[0],) or
            row['shift'].dtype != np.int32 or row['shift'].shape != (shape[0],) or
            np.any((row['shift'] < 0) | (row['shift'] > 16))):
            raise ValueError('fixed int8/int32 CNN representation required')


def weights_bytes(layers):
    validate(layers)
    return sum(value.nbytes for row in layers for value in row.values())


def activation_layout():
    height, width = 64, 32
    shapes = []
    for shape, stride in zip(SHAPES[:5], STRIDES, strict=True):
        height, width = (height + stride - 1) // stride, (width + stride - 1) // stride
        shapes.append((shape[0], height, width))
    counts = [int(np.prod(shape)) for shape in shapes]
    # Planned C ping-pong reuse. NumPy's int64 temporaries are NOT this budget.
    estimate = max(counts[::2]) + max(2048*2, *counts[1::2])
    return dict(uint8_shapes=shapes, ping_pong_bytes_estimate=estimate, target_workspace_measured=False)


def inputs(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2 or
        values.shape[1] != FEATURE_COUNT or not 1 <= len(values) <= 65536 or
        not np.isfinite(values).all() or np.any(abs(values) > 16) or
        np.any(values[:,2048:2112] > 0) or np.any((values[:,-1] < 0) | (values[:,-1] > 4))):
        raise ValueError('bounded float32 logmel/negative energy/duration vectors required')
    return values[:,:2048].reshape(-1,1,64,32), values[:,2048:]


def _conv(values, layer, stride, groups):
    weight = layer['weight'].astype(np.int64)
    kernel = weight.shape[-1]
    values = values.astype(np.int64)
    if kernel == 1:
        total = np.einsum('nchw,oc->nohw', values, weight[:,:,0,0], optimize=True)
    else:
        pad = kernel // 2
        padded = np.pad(values, ((0,0),(0,0),(pad,pad),(pad,pad)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel,kernel), axis=(2,3))[:,:,::stride,::stride]
        if groups == 1:
            total = np.einsum('nchwkl,ockl->nohw', windows, weight, optimize=True)
        else:
            total = np.einsum('nchwkl,ckl->nchw', windows, weight[:,0], optimize=True)
    total += layer['bias'].astype(np.int64)[None,:,None,None]
    total //= (1 << layer['shift'].astype(np.int64))[None,:,None,None]
    return np.clip(total, 0, 192).astype(np.uint8)


def scores(values, layers):
    validate(layers)
    images, tails = inputs(values)
    rows = []
    for start in range(0, len(values), BATCH):
        current = np.rint(images[start:start+BATCH] * 32).astype(np.int16)
        for layer, stride, groups in zip(layers[:5], STRIDES, GROUPS, strict=True):
            current = _conv(current, layer, stride, groups)
        flat = current.reshape(len(current), -1).astype(np.int64)
        tail = np.rint(tails[start:start+BATCH] * 32).astype(np.int64)
        joined = np.concatenate((flat, tail), axis=1)
        head = layers[-1]
        total = joined @ head['weight'].astype(np.int64).T + head['bias'].astype(np.int64)
        shift = head['shift'].astype(np.int64) - 5  # Q5 activations -> Q10 logits.
        for output in range(7):
            if shift[output] >= 0:
                total[:,output] //= 1 << shift[output]
            else:
                total[:,output] *= 1 << -shift[output]
        rows.append(np.clip(total, -(2**31), 2**31-1).astype(np.int32))
    return np.concatenate(rows)
