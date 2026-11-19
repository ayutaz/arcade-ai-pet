"""CPU-only quantization-aware fitting of 224 coefficients over a frozen CNN."""
import math
from time import monotonic
import numpy as np
from .dscnn_model import inputs,weights_bytes
from .dscnn_residual import (prefix_accumulators,integer_head,quantize_auxiliary,
                             make_layers,assert_frozen,frozen_digest)
from .domain_training import _inputs,validation_rank,select_threshold,shortlist


def _ste_logits(accumulators,means,parameter,shifts):
    import torch
    scale=torch.exp2(shifts)[:,None]
    bounded=torch.clamp(parameter*scale,-127,127)
    quantized=bounded+(torch.round(bounded)-bounded).detach()
    total=accumulators+means@quantized.T
    output=torch.clamp(total/torch.exp2(shifts-5),-(2**31),2**31-1)
    return (output+(torch.floor(output)-output).detach())/1024


def fit(x,y,vx,vy,groups,*,base_layers,epochs=30,batch_size=512,seed=42,
        device='cpu',max_steps=4000,max_seconds=300,envelope=True):
    import torch
    started=monotonic()
    if (envelope is not True or device!='cpu' or type(epochs) is not int or not 1<=epochs<=1000 or
        type(batch_size) is not int or not 1<=batch_size<=2048 or type(seed) is not int or not 0<=seed<2**31 or
        type(max_steps) is not int or not 1<=max_steps<=1000000 or type(max_seconds) not in (int,float) or
        not math.isfinite(max_seconds) or not 0<max_seconds<=86400):
        raise ValueError('bounded CPU-only frozen-prefix recipe required')
    before=frozen_digest(base_layers)
    _inputs(vy,groups)
    for values,labels in ((x,y),(vx,vy)):
        inputs(values,envelope=True)
        if (not isinstance(labels,np.ndarray) or labels.shape!=(len(values),) or
            not np.issubdtype(labels.dtype,np.integer) or set(labels.tolist())!=set(range(7))):
            raise ValueError('all seven aligned classes required')
    if ((len(y)+batch_size-1)//batch_size)*epochs>max_steps:
        raise ValueError('requested epochs exceed residual step budget')
    def check():
        if frozen_digest(base_layers)!=before:raise ValueError('frozen base changed during fit')
        if monotonic()-started>max_seconds:raise TimeoutError('residual deadline; no partial candidate')
    check()
    train_acc=prefix_accumulators(x,base_layers);validation_acc=prefix_accumulators(vx,base_layers)
    check()
    train_means=np.rint(x[:,2113:]*32).astype(np.int16)
    validation_means=np.rint(vx[:,2113:]*32).astype(np.int16)
    shifts=base_layers[-1]['shift']
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    parameter=torch.nn.Parameter(torch.zeros((7,32),dtype=torch.float64))
    optimizer=torch.optim.Adam([parameter],lr=.003,weight_decay=.01)
    tx=torch.tensor(train_means,dtype=torch.float64);ta=torch.tensor(train_acc,dtype=torch.float64)
    ty=torch.tensor(y.astype(np.int64));ts=torch.tensor(shifts,dtype=torch.float64)
    counts=np.bincount(y,minlength=7);class_weight=1/np.maximum(counts,1);class_weight/=class_weight.mean()
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight,dtype=torch.float64))
    upper=127/torch.exp2(ts)[:,None];lower=-upper
    export=lambda:quantize_auxiliary(parameter.detach().numpy(),shifts)
    zero=np.zeros((7,32),dtype=np.int8)
    baseline=dict(auxiliary=zero,epoch=0)
    checkpoints=[];steps=0;first_loss=last_loss=None
    for epoch in range(epochs):
        order=torch.randperm(len(y))
        for start in range(0,len(y),batch_size):
            check();indexes=order[start:start+batch_size];optimizer.zero_grad()
            loss=lossfn(_ste_logits(ta[indexes],tx[indexes],parameter,ts),ty[indexes])
            loss.backward();optimizer.step()
            with torch.no_grad():parameter.copy_(torch.maximum(lower,torch.minimum(parameter,upper)))
            last_loss=float(loss.detach())
            if not math.isfinite(last_loss):raise ValueError('nonfinite residual loss')
            if first_loss is None:first_loss=last_loss
            steps+=1;check()
        if epoch%5==0 or epoch==epochs-1:
            auxiliary=export();logits=integer_head(validation_acc,validation_means,auxiliary,shifts)
            checkpoints.append(dict(auxiliary=auxiliary,epoch=epoch+1,rank=validation_rank(vy,logits.argmax(1),groups)))
    candidates=[baseline,*shortlist(checkpoints)];best=best_key=None
    for row in candidates:
        check();logits=integer_head(validation_acc,validation_means,row['auxiliary'],shifts)
        threshold,reports,passed=select_threshold(vy,logits,groups)
        key=(passed,min(r['macro_f1'] for r in reports),sum(r['macro_f1'] for r in reports),-threshold,-row['epoch'])
        if best_key is None or key>best_key:
            best_key=key;best=dict(layers=make_layers(base_layers,row['auxiliary']),selected_epoch=row['epoch'],
                                  checkpoint_threshold_q10=threshold,selected_auxiliary_nonzero_coefficients=int(np.count_nonzero(row['auxiliary'])))
    check();assert_frozen(base_layers,best['layers'])
    best.update(seed=seed,epochs=epochs,batch_size=batch_size,training_device='cpu',training_steps=steps,
                checkpoint_candidates_evaluated=len(candidates),zero_baseline_included=True,
                first_batch_loss=first_loss,final_batch_loss=last_loss,fit_elapsed_seconds=monotonic()-started,
                weights_bytes=weights_bytes(best['layers'],envelope=True),trainable_coefficients=224,
                base_prefix_unchanged=True,base_numeric_sha256=before,
                final_auxiliary_nonzero_coefficients=int(np.count_nonzero(export())),
                selection_metric='calibrated_worst_development_domain_profile',learning_method='frozen_prefix_auxiliary_qat')
    return best
