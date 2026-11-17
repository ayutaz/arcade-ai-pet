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


def make_arrays(dataset,candidate,*,frontend='aligned'):
    if candidate not in CANDIDATES or frontend not in ('aligned','adaptive','subtract'):raise ValueError('conditioning candidate/frontend')
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
            out['x'].append(extract(level(condition(simulate(pcm,profile,seed),candidate))))
            out['y'].append(label);out['profiles'].append(profile)
    for out in arrays.values():
        if not out['y']:raise ValueError('nonempty train and validation required')
        out['x']=np.asarray(out['x'],dtype=np.float32);out['y']=np.asarray(out['y'],dtype=np.int64)
    return arrays


def train_channel_candidate(dataset,output,candidate,*,epochs=600,frontend='aligned'):
    from .training import fit,scores,decisions
    from .channel_robustness import measure
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh training output required')
    data=make_arrays(dataset,candidate,frontend=frontend)
    result=fit(data['train']['x'],data['train']['y'],data['validation']['x'],data['validation']['y'],
               epochs=epochs,hidden=64,seed=42)
    layers=result.pop('layers');v=data['validation'];logits=scores(v['x'],layers)
    options=[]
    for threshold in (0,128,256,512,768,1024,1536,2048,3072,4096):
        predictions=decisions(logits,threshold);rows=[]
        for profile in PROFILES:
            mask=np.array(v['profiles'])==profile
            rows.append(dict(candidate=candidate,profile=profile,**measure(v['y'][mask],predictions[mask])))
        gates=all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in rows)
        options.append((not gates,-min(r['macro_f1'] for r in rows),-sum(r['macro_f1'] for r in rows),threshold,rows))
    selected=min(options,key=lambda r:r[:4]);threshold=selected[3];rows=selected[4]
    result.update(candidate=candidate,preprocessing='condition-then-peak4096-v1',frontend_version=frontend,validation=rows,
                  threshold_q10=threshold,accepted_validation=not selected[0],final_evaluated=False,
                  real_voice_tested=False,device_tested=False,vast_usd=0,
                  weights_bytes=sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in layers),
                  split_examples={s:len(data[s]['y']) for s in data},
                  dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest())
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    result['weights_sha256']=hashlib.sha256((output/'weights.npz').read_bytes()).hexdigest()
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--candidate',choices=CANDIDATES,required=True);p.add_argument('--epochs',type=int,default=600)
    p.add_argument('--frontend',choices=('aligned','adaptive','subtract'),default='aligned')
    a=p.parse_args();r=train_channel_candidate(a.dataset,a.output,a.candidate,epochs=a.epochs,frontend=a.frontend)
    print(json.dumps({k:r[k] for k in ('candidate','accepted_validation','threshold_q10','weights_bytes','selected_epoch')}))
