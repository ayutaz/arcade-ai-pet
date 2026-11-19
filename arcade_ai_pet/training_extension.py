"""Append qualified new development sources to train; preserve existing evaluation bytes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata
import numpy as np
from . import neural_corpus
from .neural_corpus import validate_plan
from .data import SOURCES,read_pcm,trim_silence,augment
from .features import features,split_for
from .training import load_dataset
from .native_evaluation import _examples


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _caption(value):
    return unicodedata.normalize('NFKC',value).strip()


def append_qualified_training(base, extra, output, *, reserved_plan):
    base,extra,output,reserved_plan = [Path(p).absolute() for p in (base,extra,output,reserved_plan)]
    if (any(p.resolve()!=p for p in (base,extra,output,reserved_plan)) or output.exists() or
            any((extra/name).exists() for name in ('manifest.json','train.npz','validation.npz','test.npz'))):
        raise ValueError('real inputs, unpartitioned extension and fresh output required')
    original = json.loads((base/'manifest.json').read_text())
    reserved = json.loads(reserved_plan.read_text())
    plan = json.loads((extra/'plan.json').read_text())
    reserved_digest,digest = validate_plan(reserved),validate_plan(plan)
    if (original.get('role')!='development' or original.get('feature_version')!='aligned-mfcc-v2' or
            original.get('plan_sha256')!=reserved_digest or original.get('variants')!=plan['variants'] or
            plan['provider']!=reserved['provider']):
        raise ValueError('reviewed provider, original reserved plan and development variants required')
    data = load_dataset(base)
    final = lambda p: [r for r in p['conditions'] if r['role']=='final']
    old_captions = {_caption(r['caption']) for r in reserved['conditions']}
    old_ids = {r['id'] for r in reserved['conditions']}
    conditions = {r['id']:r for r in plan['conditions'] if r['role']=='development'}
    if (final(plan)!=final(reserved) or any(_caption(r['caption']) in old_captions or r['id'] in old_ids for r in conditions.values())):
        raise ValueError('reserved final conditions must remain unchanged; new development conditions required')
    qualified = json.loads((extra/'qualified-sources.json').read_text())
    progress = json.loads((extra/'progress.json').read_text())
    if (qualified.get('schema')!=1 or qualified.get('role')!='development' or
            qualified.get('plan_sha256')!=digest or progress.get('status')!='qualified' or progress.get('plan_sha256')!=digest or
            qualified.get('generator_sha256')!=_sha(neural_corpus.__file__) or
            qualified.get('audit_sha256')!=_sha(Path(neural_corpus.__file__).with_name('generated_audio.py'))):
        raise ValueError('completed unchanged development qualification required')
    audits = {}
    for row in progress['attempts']:
        if row.get('content_check_passed') is True:
            if row['id'] in audits:
                raise ValueError('unique successful audit identity required')
            audits[row['id']] = row
    ids = {r['id'] for r in original['records']}
    waves = {r['sha256'] for r in original['records'] if 'sha256' in r}
    added,values,labels,copies,covered = [],[],[],[],set()
    for row in qualified['records']:
        ident,label,prompt = row['id'],row['label'],row['prompt']
        audit = audits.get(ident,{})
        prefix = f"neural/{digest}/{row['condition_id']}/{label}/{neural_corpus._digest(prompt)}/"
        if (type(ident) is not str or not re.fullmatch('[0-9a-f]{64}',ident) or ident in ids or row['sha256'] in waves or
                row['condition_id'] not in conditions or prompt not in SOURCES.get(label,()) or
                not row['source'].startswith(prefix) or row['source'][len(prefix):] not in map(str,range(plan['attempts'])) or
                ident!=hashlib.sha256(row['source'].encode()).hexdigest() or row['split']!=split_for(row['source']) or
                audit.get('status')!='matched' or any(audit.get(k)!=row[v] for k,v in
                    (('wave_sha256','sha256'),('label','label'),('prompt','prompt'),('condition_id','condition_id')))):
            raise ValueError('new qualified source/audit identity differs or overlaps')
        file = extra/'sources'/(ident+'.wav')
        if file.is_symlink() or not file.is_file() or _sha(file)!=row['sha256']:
            raise ValueError('qualified extension WAV changed')
        ids.add(ident)
        waves.add(row['sha256'])
        for nonce in range(10000):
            source = f'train-extension-v1/{digest}/{ident}/{nonce}'
            if split_for(source)=='train':
                break
        else:
            raise ValueError('bounded train identity search exhausted')
        new_id = hashlib.sha256(source.encode()).hexdigest()
        if new_id in ids:
            raise ValueError('new train source collision')
        ids.add(new_id)
        added.append(dict(row,origin_id=ident,origin_source=row['source'],origin_split=row['split'],source=source,id=new_id,split='train'))
        pcm = trim_silence(read_pcm(file))
        values.extend(features(augment(pcm,int(new_id[:8],16)+i)) for i in range(plan['variants']))
        labels.extend([label]*plan['variants'])
        copies.append((file,new_id))
        covered.add(prompt)
    if covered!={text for texts in SOURCES.values() for text in texts}:
        raise ValueError('complete qualified extension prompt coverage required')
    base_copies = []
    for row in original['records']:
        if 'sha256' in row:
            file = base/'sources'/(row['id']+'.wav')
            if file.is_symlink() or not file.is_file() or _sha(file)!=row['sha256']:
                raise ValueError('base source changed')
            base_copies.append(file)
    for _ in _examples(base):
        pass
    # Only after validation: preserve the old evaluation files byte for byte.
    output.mkdir(parents=True)
    (output/'sources').mkdir()
    for file in base_copies:
        shutil.copyfile(file,output/'sources'/file.name)
    for file,ident in copies:
        shutil.copyfile(file,output/'sources'/(ident+'.wav'))
    for name in ('validation.npz','test.npz'):
        shutil.copyfile(base/name,output/name)
    np.savez_compressed(output/'train.npz',x=np.concatenate((data['train']['x'],np.asarray(values,dtype=np.float32))),
                        y=np.concatenate((data['train']['y'],np.asarray(labels,dtype=np.int64))))
    provenance = dict(plan_sha256=digest,qualified_sha256=_sha(extra/'qualified-sources.json'),
                      generator_sha256=qualified['generator_sha256'],audit_sha256=qualified['audit_sha256'],
                      provider=plan['provider'],conditions=list(conditions.values()),speech_sources=len(added))
    manifest = dict(original,records=original['records']+added,base_manifest_sha256=_sha(base/'manifest.json'),
                    development_split='evaluation-preserving-training-extension-v1',
                    training_extensions=original.get('training_extensions',[])+[provenance])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(speech_sources_added=len(added),added_training_rows=len(labels),validation_changed=False,test_changed=False,
                evaluation_sha256={name:_sha(output/name) for name in ('validation.npz','test.npz')},
                final_conditions_used=False,real_voice_tested=False,device_tested=False)
