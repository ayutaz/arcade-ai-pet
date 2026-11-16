"""Power-of-two int8 weights, int32 biases and Q10 integer activations."""
import math
import numpy as np


def quantize_layer(weights,bias):
    w=np.asarray(weights,dtype=np.float64);b=np.asarray(bias,dtype=np.float64)
    if (w.ndim!=2 or w.shape[1]>260 or w.shape[0]>64 or b.shape!=(w.shape[0],)
        or not w.size or not np.isfinite(w).all() or not np.isfinite(b).all()):
        raise ValueError('bounded finite dense layer required')
    peak=float(np.abs(w).max())
    shift=min(16,int(math.floor(math.log2(127/peak)))) if peak else 16
    if shift<0:raise ValueError('weight range')
    qbias=np.rint(b*(1024<<shift))
    if np.any(abs(qbias)>2147483647):raise ValueError('bias overflow')
    return dict(weight=np.rint(w*(1<<shift)).astype(np.int8),bias=qbias.astype(np.int32),shift=shift)


def dense(x,layer,*,relu=False):
    x=np.asarray(x)
    if x.shape!=(layer['weight'].shape[1],) or not np.issubdtype(x.dtype,np.integer) or np.any(abs(x.astype(np.int64))>16384):
        raise ValueError('bounded Q10 activation required')
    sums=layer['weight'].astype(np.int64)@x.astype(np.int64)+layer['bias'].astype(np.int64)
    result=sums>>layer['shift']
    return np.clip(result,0,6144).astype(np.int32) if relu else result.astype(np.int32)


def predict(x,layers):
    q=np.rint(np.clip(x,-16,16)*1024).astype(np.int32)
    for index,layer in enumerate(layers):q=dense(q,layer,relu=index<len(layers)-1)
    return int(q.argmax())
