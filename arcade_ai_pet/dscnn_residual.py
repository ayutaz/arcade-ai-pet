"""Exact frozen-prefix CNN arithmetic; the only added weights are 7x32 int8."""
import hashlib
import numpy as np
from . import dscnn_model as cnn


def _shifts(shifts):
    if (not isinstance(shifts,np.ndarray) or shifts.dtype!=np.int32 or shifts.shape!=(7,) or
        np.any((shifts<0)|(shifts>16))):
        raise ValueError('seven bounded fixed int32 shifts required')


def _auxiliary(values):
    if not isinstance(values,np.ndarray) or values.dtype!=np.int8 or values.shape!=(7,32):
        raise ValueError('fixed 7x32 int8 auxiliary weights required')


def frozen_digest(base):
    cnn.validate(base)
    digest=hashlib.sha256()
    for row in base:
        for field in ('weight','bias','shift'):digest.update(row[field].tobytes(order='C'))
    return digest.hexdigest()


def make_layers(base,auxiliary=None):
    cnn.validate(base)
    if auxiliary is None:auxiliary=np.zeros((7,32),dtype=np.int8)
    _auxiliary(auxiliary)
    layers=[{k:v.copy() for k,v in row.items()} for row in base]
    layers[-1]['weight']=np.concatenate((layers[-1]['weight'],auxiliary),axis=1)
    cnn.validate(layers,envelope=True)
    return layers


def assert_frozen(base,layers):
    cnn.validate(base);cnn.validate(layers,envelope=True)
    for old,new in zip(base[:5],layers[:5],strict=True):
        if any(not np.array_equal(old[k],new[k]) for k in ('weight','bias','shift')):
            raise ValueError('frozen convolution changed')
    if (not np.array_equal(base[-1]['weight'],layers[-1]['weight'][:,:3137]) or
        any(not np.array_equal(base[-1][k],layers[-1][k]) for k in ('bias','shift'))):
        raise ValueError('frozen head prefix, bias or shifts changed')


def prefix_accumulators(values,base):
    cnn.validate(base);images,_=cnn.inputs(values,envelope=True)
    rows=[]
    for start in range(0,len(values),cnn.BATCH):
        current=np.rint(images[start:start+cnn.BATCH]*32).astype(np.int16)
        for layer,stride,groups in zip(base[:5],cnn.STRIDES,cnn.GROUPS,strict=True):
            current=cnn._conv(current,layer,stride,groups)
        prefix=np.concatenate((current.reshape(len(current),-1).astype(np.int64),
                               np.rint(values[start:start+cnn.BATCH,2048:2113]*32).astype(np.int64)),axis=1)
        rows.append(prefix@base[-1]['weight'].astype(np.int64).T+base[-1]['bias'].astype(np.int64))
    return np.concatenate(rows)


def integer_head(accumulators,means,auxiliary,shifts):
    _auxiliary(auxiliary);_shifts(shifts)
    if (not isinstance(accumulators,np.ndarray) or accumulators.dtype!=np.int64 or accumulators.ndim!=2 or
        accumulators.shape[1]!=7 or not 1<=len(accumulators)<=65536 or
        np.any((accumulators<-(2**40))|(accumulators>2**40)) or
        not isinstance(means,np.ndarray) or means.dtype!=np.int16 or means.shape!=(len(accumulators),32) or
        np.any((means<-512)|(means>512))):
        raise ValueError('bounded int64 prefix and matching signed Q5 means required')
    total=accumulators+means.astype(np.int64)@auxiliary.astype(np.int64).T
    for output,shift in enumerate(shifts.astype(np.int64)-5):
        if shift>=0:total[:,output]//=1<<shift
        else:total[:,output]*=1<<-shift
    return np.clip(total,-(2**31),2**31-1).astype(np.int32)


def quantize_auxiliary(values,shifts):
    _shifts(shifts)
    if (not isinstance(values,np.ndarray) or values.dtype not in (np.dtype('float32'),np.dtype('float64')) or
        values.shape!=(7,32) or not np.isfinite(values).all() or np.any(abs(values)>1e6)):
        raise ValueError('finite bounded auxiliary real weights required')
    return np.clip(np.rint(values.astype(np.float64)*np.exp2(shifts)[:,None]),-127,127).astype(np.int8)
