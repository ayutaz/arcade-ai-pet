"""Unrounded real arithmetic for fixed quantized weights, not original float weights."""
import numpy as np
from .dscnn_model import validate, inputs, STRIDES, GROUPS, BATCH


def _conv(values, layer, stride, groups):
    weight=layer['weight'].astype(np.float64)
    kernel=weight.shape[-1]
    if kernel==1:
        total=np.einsum('nchw,oc->nohw',values,weight[:,:,0,0],optimize=True)
    else:
        pad=kernel//2
        padded=np.pad(values,((0,0),(0,0),(pad,pad),(pad,pad)))
        windows=np.lib.stride_tricks.sliding_window_view(padded,(kernel,kernel),axis=(2,3))[:,:,::stride,::stride]
        if groups==1:
            total=np.einsum('nchwkl,ockl->nohw',windows,weight,optimize=True)
        else:
            total=np.einsum('nchwkl,ckl->nchw',windows,weight[:,0],optimize=True)
    total+=layer['bias'].astype(np.float64)[None,:,None,None]/32
    total/=np.exp2(layer['shift'].astype(np.float64))[None,:,None,None]
    return np.clip(total,0,6)


def scores(values,layers):
    """Return float64 logits in Q10 units, without input/activation/head rounding."""
    validate(layers)
    images,tails=inputs(values)
    rows=[]
    for start in range(0,len(values),BATCH):
        current=images[start:start+BATCH].astype(np.float64)
        for layer,stride,groups in zip(layers[:5],STRIDES,GROUPS,strict=True):
            current=_conv(current,layer,stride,groups)
        joined=np.concatenate((current.reshape(len(current),-1),tails[start:start+BATCH].astype(np.float64)),axis=1)
        head=layers[-1]
        total=joined@head['weight'].astype(np.float64).T+head['bias'].astype(np.float64)/32
        rows.append(total/np.exp2(head['shift'].astype(np.float64))*1024)
    return np.concatenate(rows)
