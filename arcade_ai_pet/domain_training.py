"""Bounded domain-balanced training; selection never consumes final conditions."""
import hashlib,json
from pathlib import Path
import numpy as np
from .channel_training import make_arrays
from .channel_robustness import PROFILES,measure
from .channel_freeze import _rows,_accepted
from .training import scores,decisions
from .quantize import quantize_layer,MAX_HIDDEN


def _inputs(labels,groups):
    if (not isinstance(labels,np.ndarray) or labels.ndim!=1 or not len(labels) or
            not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>6)) or
            not isinstance(groups,np.ndarray) or groups.shape!=labels.shape or
            not np.issubdtype(groups.dtype,np.integer) or set(groups)!=set(range(16))):
        raise ValueError('sixteen complete development domain/profile groups required')


def _reports(labels,predictions,groups):
    return [dict(domain='legacy' if index<8 else 'neural',profile=PROFILES[index%8],
                 **measure(labels[groups==index],predictions[groups==index])) for index in range(16)]


def _passed(rows):
    return len(rows)==16 and all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in rows)


def validation_rank(labels,predictions,groups):
    _inputs(labels,groups)
    if (not isinstance(predictions,np.ndarray) or predictions.shape!=labels.shape or
            not np.issubdtype(predictions.dtype,np.integer) or np.any((predictions<0)|(predictions>6))):
        raise ValueError('bounded validation predictions required')
    rows=_reports(labels,predictions,groups)
    return _passed(rows),min(r['macro_f1'] for r in rows),sum(r['macro_f1'] for r in rows)/16


def select_threshold(labels,logits,groups):
    _inputs(labels,groups)
    if (not isinstance(logits,np.ndarray) or logits.shape!=(len(labels),7) or
            not np.issubdtype(logits.dtype,np.integer) or np.any((logits<-(2**31))|(logits>2**31-1))):
        raise ValueError('bounded integer validation logits required')
    logits=logits.astype(np.int64);ordered=np.sort(logits,axis=1);margins=ordered[:,-1]-ordered[:,-2]
    thresholds=sorted({0}|{int(m)+1 for m,b in zip(margins,logits.argmax(1),strict=True) if b<5 and m<4096})
    options=[]
    for threshold in thresholds:
        rows=_reports(labels,decisions(logits,threshold),groups)
        options.append((not _passed(rows),-min(r['macro_f1'] for r in rows),-sum(r['macro_f1'] for r in rows),threshold,rows))
    selected=min(options,key=lambda r:r[:4])
    return selected[3],selected[4],not selected[0]


def shortlist(checkpoints):
    if (type(checkpoints) is not list or not 1<=len(checkpoints)<=201 or
            any(type(c.get('epoch')) is not int or not 1<=c['epoch']<=1000 for c in checkpoints) or
            len({c['epoch'] for c in checkpoints})!=len(checkpoints)):
        raise ValueError('bounded unique training checkpoints required')
    key=lambda c:(c['rank'],-c['epoch'])
    selected={c['epoch']:c for c in sorted(checkpoints,key=key,reverse=True)[:8]}
    maximum=max(c['epoch'] for c in checkpoints)
    for period in range(8):
        candidates=[c for c in checkpoints if min(7,(c['epoch']-1)*8//maximum)==period]
        if candidates:
            chosen=max(candidates,key=key);selected[chosen['epoch']]=chosen
    return [selected[epoch] for epoch in sorted(selected)]


def select_checkpoint(checkpoints,values,labels,groups):
    _inputs(labels,groups)
    if type(checkpoints) is not list or not 1<=len(checkpoints)<=16:
        raise ValueError('at most sixteen calibration candidates required')
    best=None;best_key=None
    for checkpoint in checkpoints:
        logits=scores(values,checkpoint['layers']);threshold,rows,passed=select_threshold(labels,logits,groups)
        key=(passed,min(r['macro_f1'] for r in rows),sum(r['macro_f1'] for r in rows),-threshold,-checkpoint['epoch'])
        if best_key is None or key>best_key:
            best_key=key;best=dict(layers=checkpoint['layers'],selected_epoch=checkpoint['epoch'],
                threshold_q10=threshold,accepted_all_domains=passed,domain_profiles=rows,
                raw_validation_rank=validation_rank(labels,logits.argmax(1),groups),
                validation_accuracy=float(np.mean(logits.argmax(1)==labels)))
    return best


def fit(x,y,vx,vy,groups,*,epochs=1000,hidden=88,seed=42):
    import torch
    _inputs(vy,groups)
    for values,labels in ((x,y),(vx,vy)):
        if (not isinstance(values,np.ndarray) or values.ndim!=2 or values.shape[1]!=260 or not len(values) or
                labels.shape!=(len(values),) or not np.isfinite(values).all() or np.any(abs(values)>16) or
                not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>6))):
            raise ValueError('bounded development features required')
    if type(epochs) is not int or not 1<=epochs<=1000 or type(hidden) is not int or not 1<=hidden<=MAX_HIDDEN:
        raise ValueError('bounded training resource configuration required')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=torch.nn.Sequential(torch.nn.Linear(260,hidden),torch.nn.ReLU6(),torch.nn.Linear(hidden,7))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    counts=np.bincount(y,minlength=7);weight=1/np.maximum(counts,1);weight/=weight.mean()
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32))
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.long);checkpoints=[]
    for epoch in range(epochs):
        optimizer.zero_grad();loss=lossfn(model(tx),ty);loss.backward();optimizer.step()
        if epoch%5==0 or epoch==epochs-1:
            layers=[quantize_layer(model[i].weight.detach().numpy(),model[i].bias.detach().numpy()) for i in (0,2)]
            predictions=scores(vx,layers).argmax(1);rank=validation_rank(vy,predictions,groups)
            checkpoints.append(dict(layers=layers,epoch=epoch+1,rank=rank))
    candidates=shortlist(checkpoints);selected=select_checkpoint(candidates,vx,vy,groups)
    return dict(layers=selected['layers'],selected_epoch=selected['selected_epoch'],validation_accuracy=selected['validation_accuracy'],seed=seed,epochs=epochs,
                selection_metric='calibrated_worst_development_domain_profile',raw_validation_rank=selected['raw_validation_rank'],
                checkpoint_candidates_evaluated=len(candidates),checkpoint_threshold_q10=selected['threshold_q10'])


def train_candidate(domains,mixed,output,*,epochs=1000,hidden=88,neural_weight=4):
    output=Path(output).absolute();mixed=Path(mixed).absolute()
    if (output.resolve()!=output or output.exists() or type(domains) is not dict or set(domains)!={'legacy','neural'} or
            type(neural_weight) is not int or not 1<=neural_weight<=8):
        raise ValueError('fresh bounded development experiment required')
    domains={name:Path(domains[name]) for name in ('legacy','neural')};records=[];seen=set();waves=set();variants=set()
    for path in domains.values():
        manifest=json.loads((path/'manifest.json').read_text());variants.add(manifest['variants'])
        if manifest.get('role','development')!='development':raise ValueError('final training data forbidden')
        for row in manifest['records']:
            if row['id'] in seen or 'sha256' in row and row['sha256'] in waves:raise ValueError('development domain overlap')
            seen.add(row['id'])
            if 'sha256' in row:waves.add(row['sha256'])
        records.extend(manifest['records'])
    combined=json.loads((mixed/'manifest.json').read_text())
    if len(variants)!=1 or combined.get('role')!='development' or combined['records']!=records or combined['variants'] not in variants:
        raise ValueError('mixed development identity differs from component domains')
    arrays={name:make_arrays(path,'hp160',frontend='adaptive',noise_replicas=5,noise_mode='fresh',hum_replicas=3) for name,path in domains.items()}
    x=np.concatenate([arrays['legacy']['train']['x'],np.tile(arrays['neural']['train']['x'],(neural_weight,1))])
    y=np.concatenate([arrays['legacy']['train']['y'],np.tile(arrays['neural']['train']['y'],neural_weight)])
    vx=np.concatenate([a['validation']['x'] for a in arrays.values()]);vy=np.concatenate([a['validation']['y'] for a in arrays.values()])
    profiles=[p for a in arrays.values() for p in a['validation']['profiles']]
    groups=np.array([domain*8+PROFILES.index(p) for domain,a in enumerate(arrays.values()) for p in a['validation']['profiles']])
    result=fit(x,y,vx,vy,groups,epochs=epochs,hidden=hidden);layers=result.pop('layers');logits=scores(vx,layers)
    threshold,rows,accepted=select_threshold(vy,logits,groups);combined_rows=_rows(vy,logits,profiles,'hp160',threshold)
    result.update(candidate='hp160',frontend_version='adaptive',threshold_q10=threshold,validation=combined_rows,
                  domain_profiles=rows,accepted_all_domains=accepted,accepted_validation=accepted and _accepted(combined_rows),
                  training_domain_weights=dict(legacy=1,neural=neural_weight),noise_replicas=5,hum_replicas=3,noise_mode='fresh',
                  dataset_manifest_sha256=hashlib.sha256((mixed/'manifest.json').read_bytes()).hexdigest(),
                  trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  weights_bytes=sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in layers),
                  split_examples=dict(train=len(y),validation=len(vy)),final_evaluated=False,real_voice_tested=False,device_tested=False,vast_usd=0)
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    result['weights_sha256']=hashlib.sha256((output/'weights.npz').read_bytes()).hexdigest()
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
