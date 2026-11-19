"""Explicit 261-input dense quantization; legacy 260-input guard stays intact."""
import numpy as np
from .quantize import quantize_layer as previous, MAX_HIDDEN


def quantize_layer(weights,bias):
    if np.iscomplexobj(weights) or np.iscomplexobj(bias):
        raise ValueError('real duration layer required')
    w=np.asarray(weights,dtype=np.float64);b=np.asarray(bias,dtype=np.float64)
    if (w.ndim!=2 or w.shape[1]!=261 or not 1<=w.shape[0]<=MAX_HIDDEN or
            b.shape!=(w.shape[0],) or not np.isfinite(w).all() or not np.isfinite(b).all()):
        raise ValueError('bounded finite 261-input layer required')
    # Reuse the original power-of-two calibration on the full matrix peak.
    shift=previous(np.array([[float(abs(w).max())]]),np.zeros(1))['shift']
    qbias=np.rint(b*(1024<<shift))
    if np.any(abs(qbias)>2147483647):raise ValueError('duration bias overflow')
    return dict(weight=np.rint(w*(1<<shift)).astype(np.int8),bias=qbias.astype(np.int32),shift=shift)
