"""Development-only class-wise margin calibration; fixed weights and bounded search."""
import numpy as np
from .domain_training import _inputs, _reports, _passed
from .channel_robustness import PROFILES, measure
from .channel_freeze import _accepted


def _parts(logits):
    if (not isinstance(logits,np.ndarray) or logits.ndim!=2 or logits.shape[1]!=7 or not len(logits) or
            not np.issubdtype(logits.dtype,np.integer) or
            np.any((logits<-(2**31))|(logits>2**31-1))):
        raise ValueError('nonempty bounded integer seven-class logits required')
    values=logits.astype(np.int64);ordered=np.sort(values,axis=1)
    return values.argmax(1),ordered[:,-1]-ordered[:,-2]


def _table(thresholds):
    if (not isinstance(thresholds,np.ndarray) or thresholds.shape!=(5,) or
            not np.issubdtype(thresholds.dtype,np.integer) or np.any((thresholds<0)|(thresholds>4096))):
        raise ValueError('five bounded integer command thresholds required')
    return thresholds.astype(np.int64)


def _decide(best,margins,thresholds):
    return np.where((best<5)&(margins<thresholds[np.minimum(best,4)]),5,best)


def decisions(logits,thresholds):
    best,margins=_parts(logits)
    return _decide(best,margins,_table(thresholds))


def _rank(labels,predictions,groups,thresholds):
    confusion=np.bincount(groups*49+labels*7+predictions,minlength=16*49).reshape(16,7,7)
    # Both the 16 domain/profile groups and their eight pooled profiles are gates.
    confusion=np.concatenate((confusion,confusion[:8]+confusion[8:]))
    tp=np.diagonal(confusion,axis1=1,axis2=2)
    denominator=confusion.sum(1)+confusion.sum(2)
    f1=np.divide(2*tp,denominator,out=np.zeros((24,7),dtype=float),where=denominator!=0).mean(1)
    false=confusion[:,5:,:5].sum((1,2))/np.maximum(1,confusion[:,5:,:].sum((1,2)))
    recall=tp[:,:5].sum(1)/np.maximum(1,confusion[:,:5,:].sum((1,2)))
    accepted=bool(np.all((f1>=.90)&(false<=.05)&(recall>=.80)))
    return (accepted,float(f1.min()),float(f1.mean()),float(recall.min()),-float(false.max()),
            -int(thresholds.sum()),*map(lambda x:-int(x),thresholds))


def calibrate(labels,logits,groups,*,initial_threshold=0,max_passes=4):
    _inputs(labels,groups)
    labels=labels.astype(np.int64);groups=groups.astype(np.int64)
    if (type(initial_threshold) is not int or not 0<=initial_threshold<=4096 or
            type(max_passes) is not int or not 1<=max_passes<=8):
        raise ValueError('bounded development calibration required')
    best,margins=_parts(logits)
    if len(best)!=len(labels):raise ValueError('matching development labels/logits required')
    # A decision changes only at an integer margin + 1. No word/source-specific rule.
    choices=[sorted({0}|{int(m)+1 for m in margins[best==c] if m<4096}) for c in range(5)]
    starts=[np.full(5,initial_threshold,dtype=np.int64)]
    if initial_threshold:starts.append(np.zeros(5,dtype=np.int64))
    cache={};selected=None;selected_key=None;passes=[]
    def rank(table):
        key=tuple(map(int,table))
        if key not in cache:cache[key]=_rank(labels,_decide(best,margins,table),groups,table)
        return cache[key]
    for start in starts:
        current=start.copy();current_key=rank(current);completed=0
        for _ in range(max_passes):
            before=current.copy();completed+=1
            for command in range(5):
                chosen=current;chosen_key=current_key
                for value in choices[command]:
                    trial=current.copy();trial[command]=value;trial_key=rank(trial)
                    if trial_key>chosen_key:chosen,chosen_key=trial,trial_key
                current,current_key=chosen,chosen_key
            if np.array_equal(before,current):break
        passes.append(completed)
        if selected_key is None or current_key>selected_key:
            selected,selected_key=current.copy(),current_key
    predictions=_decide(best,margins,selected)
    rows=_reports(labels,predictions,groups)
    pooled=[dict(profile=profile,**measure(labels[groups%8==index],predictions[groups%8==index]))
            for index,profile in enumerate(PROFILES)]
    return dict(thresholds_q10=selected.tolist(),domain_profiles=rows,pooled_profiles=pooled,
                accepted_all_domains=_passed(rows),accepted_validation=_passed(rows) and _accepted(pooled),
                evaluations=len(cache),passes_per_start=passes,search='bounded_coordinate_class_margins_v1',
                global_optimality_proven=False,final_evaluated=False,device_tested=False)
