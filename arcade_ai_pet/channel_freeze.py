"""Freeze channel preprocessing and model before held-out synthetic evaluation."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from .channel_training import make_arrays,level
from .channel_robustness import CANDIDATES,PROFILES,condition,simulate,measure
from .training import scores,decisions
from .frozen import read_layers
from .native_evaluation import _examples

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('uv.lock','.python-version','arcade_ai_pet/channel_freeze.py',
         'arcade_ai_pet/channel_training.py','arcade_ai_pet/channel_frontend.py',
         'arcade_ai_pet/channel_robustness.py','arcade_ai_pet/features.py',
         'arcade_ai_pet/data.py','arcade_ai_pet/native_evaluation.py',
         'arcade_ai_pet/training.py','arcade_ai_pet/quantize.py','arcade_ai_pet/frozen.py','native/conditioning.c')


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _source_hashes():return {name:_hash(ROOT/name) for name in SOURCES}
def _accepted(rows):return bool(len(rows)==8 and all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in rows))


def _rows(labels,logits,profiles,candidate,threshold):
    predictions=decisions(logits,threshold);profiles=np.array(profiles)
    return [dict(candidate=candidate,profile=p,**measure(np.asarray(labels)[profiles==p],predictions[profiles==p])) for p in PROFILES]


def freeze_channel(checkpoint,development,output):
    checkpoint=Path(checkpoint).resolve();development=Path(development).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh frozen output required')
    meta=json.loads((checkpoint/'validation.json').read_text())
    layers,size=read_layers(checkpoint/'weights.npz');digest=_hash(checkpoint/'weights.npz')
    if digest!=meta['weights_sha256'] or _hash(development/'manifest.json')!=meta['dataset_manifest_sha256']:
        raise ValueError('model/development identity changed')
    threshold=meta['threshold_q10'];frontend=meta['frontend_version'];candidate=meta['candidate']
    if type(threshold) is not int or not 0<=threshold<=4096 or frontend not in ('aligned','adaptive','subtract') or candidate not in CANDIDATES:
        raise ValueError('bounded fixed channel contract required')
    v=make_arrays(development,candidate,frontend=frontend)['validation']
    rows=_rows(v['y'],scores(v['x'],layers),v['profiles'],candidate,threshold)
    if not _accepted(rows):raise ValueError('independent validation recomputation failed')
    manifest=json.loads((development/'manifest.json').read_text())
    result=dict(schema=1,conditioning=candidate,frontend=frontend,peak=4096,samples=32000,rate=16000,
                threshold_q10=threshold,weights_sha256=digest,weights_bytes=size,
                architecture=[260,layers[0]['weight'].shape[0],7],source_sha256=_source_hashes(),
                development_manifest_sha256=_hash(development/'manifest.json'),
                validation_sha256=_hash(checkpoint/'validation.json'),validation=rows,
                source_ids=sorted({r['id'] for r in manifest['records']}),
                wave_hashes=sorted({r['sha256'] for r in manifest['records'] if 'sha256' in r}),
                final_evaluated=False,device_tested=False,real_voice_tested=False,
                license='CC-BY-3.0 derived synthetic model; see docs/data-license.md')
    output.mkdir(parents=True);shutil.copyfile(checkpoint/'weights.npz',output/'weights.npz')
    (output/'channel-frozen.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def _verify(frozen):
    meta=json.loads((frozen/'channel-frozen.json').read_text())
    if (meta.get('schema')!=1 or meta.get('samples')!=32000 or meta.get('rate')!=16000 or meta.get('peak')!=4096 or
        meta.get('conditioning') not in CANDIDATES or meta.get('frontend') not in ('aligned','adaptive','subtract') or
        type(meta.get('threshold_q10')) is not int or not 0<=meta['threshold_q10']<=4096):
        raise ValueError('fixed channel contract changed')
    if meta.get('source_sha256')!=_source_hashes():raise ValueError('frozen code identity changed')
    layers,size=read_layers(frozen/'weights.npz')
    if _hash(frozen/'weights.npz')!=meta['weights_sha256'] or size!=meta['weights_bytes']:
        raise ValueError('frozen weights changed')
    return meta,layers


def _channel_features(pcm,profile,seed,meta):
    pcm=level(condition(simulate(pcm,profile,seed),meta['conditioning']))
    if meta['frontend']=='aligned':
        from .features import features
        return features(pcm)
    from .channel_frontend import features
    return features(pcm,denoise=meta['frontend']=='subtract')


def evaluate_channel_frozen(frozen,dataset,output):
    frozen=Path(frozen).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh final output required')
    meta,layers=_verify(frozen);manifest=json.loads((dataset/'manifest.json').read_text())
    source_ids={r['id'] for r in manifest['records']};waves={r['sha256'] for r in manifest['records'] if 'sha256' in r}
    if not source_ids or source_ids & set(meta['source_ids']) or waves & set(meta['wave_hashes']):
        raise ValueError('final source/wave overlap or empty corpus')
    ids=[r['id'] for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
    xs={p:[] for p in PROFILES};labels=[]
    for (pcm,label,_),ident in zip(_examples(dataset),ids,strict=True):
        labels.append(label);seed=int(ident[:16],16)
        for profile in PROFILES:xs[profile].append(_channel_features(pcm,profile,seed,meta))
    rows=[]
    for profile,values in xs.items():
        logits=scores(np.asarray(values,dtype=np.float32),layers)
        rows.append(dict(candidate=meta['conditioning'],profile=profile,**measure(labels,decisions(logits,meta['threshold_q10']))))
    result=dict(schema=1,scope='frozen_synthetic_holdout_channel_stress_host_only',
                weights_sha256=meta['weights_sha256'],weights_bytes=meta['weights_bytes'],
                frozen_sha256=_hash(frozen/'channel-frozen.json'),dataset_manifest_sha256=_hash(dataset/'manifest.json'),
                source_sha256=meta['source_sha256'],threshold_q10=meta['threshold_q10'],profiles=rows,
                source_overlap=0,wave_overlap=0,accepted_synthetic=_accepted(rows),
                final_evaluated=True,model_selection_used_final=False,device_tested=False,real_voice_tested=False,vast_usd=0)
    output.mkdir(parents=True);(output/'channel-final-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
