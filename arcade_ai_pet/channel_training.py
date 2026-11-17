"""CPU-only channel augmentation experiments. Does not deploy or read device PCM."""
import hashlib,json
from pathlib import Path
import numpy as np
from .channel_robustness import CANDIDATES,PROFILES,condition,simulate
from .native_evaluation import _examples
from .features import features


def level(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,):
        raise ValueError('exactly 32000 PCM16 samples required')
    values=pcm.astype(np.int64);peak=int(abs(values).max())
    if not peak:return pcm.copy()
    # Integer division toward zero, including -32768, no new information implied.
    scaled=values*4096
    return (np.sign(scaled)*(abs(scaled)//peak)).astype(np.int16)


def _noise_options(replicas,mode,hum_replicas=1):
    if any(type(n) is not int or not 1<=n<=8 for n in (replicas,hum_replicas)) or mode not in ('fresh','repeat'):
        raise ValueError('bounded training noise options required')


def make_arrays(dataset,candidate,*,frontend='aligned',noise_replicas=1,noise_mode='fresh',hum_replicas=1):
    if candidate not in CANDIDATES or frontend not in ('aligned','adaptive','subtract'):raise ValueError('conditioning candidate/frontend')
    _noise_options(noise_replicas,noise_mode,hum_replicas)
    extract=features
    if frontend!='aligned':
        from .channel_frontend import features as adaptive
        extract=lambda pcm:adaptive(pcm,denoise=frontend=='subtract')
    dataset=Path(dataset);manifest=json.loads((dataset/'manifest.json').read_text())
    expanded=[(row['split'],row['id']) for row in manifest['records']
              for _ in range(manifest['variants'] if 'sha256' in row else 1)]
    arrays={split:dict(x=[],y=[],profiles=[]) for split in ('train','validation')}
    for (pcm,label,_),(split,ident) in zip(_examples(dataset),expanded,strict=True):
        if split not in arrays:continue
        seed=int(ident[:16],16);out=arrays[split]
        for profile in PROFILES:
            replicas=(noise_replicas if profile=='noise' else hum_replicas if profile=='hum' else 1) if split=='train' else 1
            for replica in range(replicas):
                current=seed
                if replica and profile=='noise' and noise_mode=='fresh':
                    current=int.from_bytes(hashlib.sha256(ident.encode()+pcm.tobytes()+bytes([replica])).digest()[:8],'little')
                out['x'].append(extract(level(condition(simulate(pcm,profile,current),candidate))))
                out['y'].append(label);out['profiles'].append(profile)
    for out in arrays.values():
        if not out['y']:raise ValueError('nonempty train and validation required')
        out['x']=np.asarray(out['x'],dtype=np.float32);out['y']=np.asarray(out['y'],dtype=np.int64)
    return arrays


def select_channel_threshold(labels,logits,profiles,candidate):
    from .training import decisions
    from .channel_robustness import measure
    if (candidate not in CANDIDATES or not isinstance(logits,np.ndarray) or logits.shape!=(len(labels),7) or
        not np.issubdtype(logits.dtype,np.integer) or np.any((logits<-(2**31))|(logits>2**31-1)) or
        len(profiles)!=len(labels) or set(profiles)!=set(PROFILES)):
        raise ValueError('bounded channel validation logits/profiles required')
    ordered=np.sort(logits.astype(np.int64),axis=1);margins=ordered[:,-1]-ordered[:,-2]
    # With integer margins and rejection iff margin<threshold, only m+1 can
    # change a decision. Cover all allowed thresholds 0..4096, without test data.
    thresholds=sorted({0}|{int(m)+1 for m,b in zip(margins,logits.argmax(axis=1),strict=True) if b<5 and m<4096})
    options=[];profile_array=np.array(profiles)
    for threshold in thresholds:
        predictions=decisions(logits,threshold);rows=[]
        for profile in PROFILES:
            mask=profile_array==profile
            rows.append(dict(candidate=candidate,profile=profile,**measure(np.asarray(labels)[mask],predictions[mask])))
        gates=all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in rows)
        options.append((not gates,-min(r['macro_f1'] for r in rows),-sum(r['macro_f1'] for r in rows),threshold,rows))
    selected=min(options,key=lambda r:r[:4])
    return selected[3],selected[4],not selected[0]


def train_channel_candidate(dataset,output,candidate,*,epochs=600,frontend='aligned',hidden=64,selection='overall',noise_replicas=1,noise_mode='fresh',hum_replicas=1):
    from .training import fit,scores,decisions,MAX_HIDDEN
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh training output required')
    if type(hidden) is not int or not 1<=hidden<=MAX_HIDDEN or type(epochs) is not int or not 1<=epochs<=1000:
        raise ValueError('training resource bounds')
    if selection not in ('overall','worst'):raise ValueError('checkpoint selection')
    _noise_options(noise_replicas,noise_mode,hum_replicas)
    data=make_arrays(dataset,candidate,frontend=frontend,noise_replicas=noise_replicas,noise_mode=noise_mode,hum_replicas=hum_replicas)
    groups=np.array([PROFILES.index(p) for p in data['validation']['profiles']]) if selection=='worst' else None
    result=fit(data['train']['x'],data['train']['y'],data['validation']['x'],data['validation']['y'],
               epochs=epochs,hidden=hidden,seed=42,validation_groups=groups)
    layers=result.pop('layers');v=data['validation'];logits=scores(v['x'],layers)
    threshold,rows,accepted=select_channel_threshold(v['y'],logits,v['profiles'],candidate)
    result.update(candidate=candidate,preprocessing='condition-then-peak4096-v1',frontend_version=frontend,
                  noise_replicas=noise_replicas,noise_mode=noise_mode,hum_replicas=hum_replicas,validation=rows,
                  threshold_q10=threshold,threshold_search='integer-decision-boundaries-v1',accepted_validation=accepted,final_evaluated=False,
                  real_voice_tested=False,device_tested=False,vast_usd=0,
                  weights_bytes=sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in layers),
                  split_examples={s:len(data[s]['y']) for s in data},
                  dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest())
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    result['weights_sha256']=hashlib.sha256((output/'weights.npz').read_bytes()).hexdigest()
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def reselect_threshold(checkpoint,dataset,output):
    """Validation-only integer boundary search; never overwrite an old result."""
    import shutil
    from .frozen import read_layers
    from .training import scores
    checkpoint=Path(checkpoint).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh reselection output required')
    metadata=json.loads((checkpoint/'validation.json').read_text())
    digest=hashlib.sha256((checkpoint/'weights.npz').read_bytes()).hexdigest()
    if digest!=metadata['weights_sha256']:raise ValueError('weights identity changed')
    if hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()!=metadata['dataset_manifest_sha256']:
        raise ValueError('development manifest identity changed')
    layers,_=read_layers(checkpoint/'weights.npz')
    v=make_arrays(dataset,metadata['candidate'],frontend=metadata['frontend_version'])['validation']
    threshold,rows,accepted=select_channel_threshold(v['y'],scores(v['x'],layers),v['profiles'],metadata['candidate'])
    metadata.update(threshold_q10=threshold,validation=rows,accepted_validation=accepted,
                    threshold_search='integer-decision-boundaries-v1',final_evaluated=False,
                    device_tested=False,real_voice_tested=False,
                    previous_validation_sha256=hashlib.sha256((checkpoint/'validation.json').read_bytes()).hexdigest())
    output.mkdir(parents=True);shutil.copyfile(checkpoint/'weights.npz',output/'weights.npz')
    (output/'validation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--candidate',choices=CANDIDATES,required=True);p.add_argument('--epochs',type=int,default=600)
    p.add_argument('--frontend',choices=('aligned','adaptive','subtract'),default='aligned')
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--selection',choices=('overall','worst'),default='overall')
    p.add_argument('--noise-replicas',type=int,default=1);p.add_argument('--noise-mode',choices=('fresh','repeat'),default='fresh')
    p.add_argument('--hum-replicas',type=int,default=1)
    a=p.parse_args();r=train_channel_candidate(a.dataset,a.output,a.candidate,epochs=a.epochs,frontend=a.frontend,hidden=a.hidden,selection=a.selection,noise_replicas=a.noise_replicas,noise_mode=a.noise_mode,hum_replicas=a.hum_replicas)
    print(json.dumps({k:r[k] for k in ('candidate','accepted_validation','threshold_q10','weights_bytes','selected_epoch')}))
