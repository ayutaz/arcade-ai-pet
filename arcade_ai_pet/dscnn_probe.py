"""Opt-in host probes of original float checkpoints; no selection, files or device I/O."""
import hashlib
import json
import numpy as np
from .dscnn_model import SHAPES,validate
from .dscnn_quantize import fold_batch_norm,quantize_layer
from .domain_training import _inputs

BN_EPS=1e-5


def _state_shapes(envelope,time_pool):
    if type(envelope) is not bool or type(time_pool) is not bool or time_pool and not envelope:
        raise ValueError('explicit compatible checkpoint modes required')
    fields={}
    for i,shape in enumerate(SHAPES[:5]):
        fields[f'blocks.{i}.0.weight']=(shape,np.dtype('float32'))
        for name in ('weight','bias','running_mean','running_var'):
            fields[f'blocks.{i}.1.{name}']=((shape[0],),np.dtype('float32'))
        fields[f'blocks.{i}.1.num_batches_tracked']=((),np.dtype('int64'))
    fields['head.weight']=((7,865 if time_pool else 3169 if envelope else 3137),np.dtype('float32'))
    fields['head.bias']=((7,),np.dtype('float32'))
    return fields


def validate_state(state,*,envelope=False,time_pool=False):
    expected=_state_shapes(envelope,time_pool)
    if type(state) is not dict or set(state)!=set(expected):raise ValueError('exact fixed float state fields required')
    for key,(shape,dtype) in expected.items():
        value=state[key]
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype:
            raise ValueError('fixed float32/int64 checkpoint representation required')
        if dtype==np.dtype('float32'):
            if not np.isfinite(value).all() or np.any(abs(value)>1e6) or key.endswith('running_var') and np.any(value<0):
                raise ValueError('finite bounded float state and nonnegative BN variance required')
        elif not 0<=int(value)<=1000000000:raise ValueError('bounded BN batch count required')


def state_digest(state,*,envelope=False,time_pool=False):
    validate_state(state,envelope=envelope,time_pool=time_pool)
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key];h.update(json.dumps([key,value.dtype.str,value.shape],separators=(',',':')).encode());h.update(value.tobytes(order='C'))
    return h.hexdigest()


def quantized_state(state,*,envelope=False,time_pool=False):
    validate_state(state,envelope=envelope,time_pool=time_pool);layers=[]
    for i,shape in enumerate(SHAPES[:5]):
        stem=f'blocks.{i}.1.'
        weight,bias=fold_batch_norm(state[f'blocks.{i}.0.weight'],np.zeros(shape[0]),
            state[stem+'running_mean'],state[stem+'running_var'],state[stem+'weight'],state[stem+'bias'],eps=BN_EPS)
        layers.append(quantize_layer(weight,bias))
    layers.append(quantize_layer(state['head.weight'],state['head.bias']))
    validate(layers,envelope=envelope,time_pool=time_pool)
    return layers


def _float_logits(values):
    if (not isinstance(values,np.ndarray) or values.dtype!=np.float32 or values.ndim!=2 or
        values.shape[1]!=7 or not 1<=len(values)<=65536 or not np.isfinite(values).all() or np.any(abs(values)>1e6)):
        raise ValueError('bounded finite float32 raw logits required')


def logits_digest(values):
    _float_logits(values)
    return hashlib.sha256(values.tobytes(order='C')).hexdigest()


def compare(labels,float_logits,integer_logits,groups):
    _inputs(labels,groups);_float_logits(float_logits)
    if (float_logits.shape!=(len(labels),7) or not isinstance(integer_logits,np.ndarray) or
        integer_logits.dtype!=np.int32 or integer_logits.shape!=float_logits.shape):
        raise ValueError('matching fixed int32 Q10 and float32 raw logits required')
    floating=float_logits.argmax(1);integer=integer_logits.argmax(1)
    fc=floating==labels;ic=integer==labels
    def counts(mask):
        count=lambda values:int(np.count_nonzero(values&mask))
        return dict(examples=int(np.count_nonzero(mask)),float_correct=count(fc),integer_correct=count(ic),
            both_correct=count(fc&ic),float_only_correct=count(fc&~ic),integer_only_correct=count(~fc&ic),
            both_wrong=count(~fc&~ic),raw_agreement=count(floating==integer))
    rows=[dict(group=g,totals=counts(groups==g),by_class=[dict(label=y,**counts((groups==g)&(labels==y))) for y in range(7)]) for g in range(16)]
    delta=abs(float_logits.astype(np.float64)*1024-integer_logits.astype(np.float64))
    return dict(scope='development_float_vs_integer_raw_top1_not_quality_gate',totals=counts(np.ones(len(labels),dtype=bool)),
                groups=rows,maximum_abs_logit_difference_q10=float(delta.max()),mean_abs_logit_difference_q10=float(delta.mean()))


def capture_state(model,*,envelope=False,time_pool=False):
    if (getattr(model,'training',True) or not hasattr(model,'blocks') or len(model.blocks)!=5 or
        any(getattr(block[1],'eps',None)!=BN_EPS for block in model.blocks)):
        raise ValueError('trusted eval CNN with fixed BN epsilon required')
    state={key:value.detach().cpu().numpy().copy() for key,value in model.state_dict().items()}
    validate_state(state,envelope=envelope,time_pool=time_pool)
    return state


def restore_state(state,*,envelope=False,time_pool=False,device='cpu'):
    validate_state(state,envelope=envelope,time_pool=time_pool)
    if type(device) is not str or device not in ('cpu','mps'):raise ValueError('local float checkpoint backend required')
    import torch
    from .dscnn_training import build_model
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    with torch.random.fork_rng(devices=[]):
        model=build_model(envelope=envelope,time_pool=time_pool)
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in state.items()},strict=True)
    return model.to(device).eval()


def float_scores(model,values,*,envelope=False,time_pool=False,device='cpu'):
    from .dscnn_model import inputs
    inputs(values,envelope=envelope)
    capture_state(model,envelope=envelope,time_pool=time_pool)
    if type(device) is not str or device not in ('cpu','mps') or any(p.device.type!=device for p in model.parameters()):
        raise ValueError('declared matching local eval backend required')
    import torch
    rows=[]
    with torch.no_grad():
        for start in range(0,len(values),256):
            tensor=torch.from_numpy(np.ascontiguousarray(values[start:start+256])).to(device)
            rows.append(model(tensor).detach().cpu().numpy().copy())
    result=np.concatenate(rows);_float_logits(result)
    return result
