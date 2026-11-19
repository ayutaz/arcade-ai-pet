"""Version a development-only corpus before training with source-level class coverage."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from .features import split_for,features
from .data import read_pcm,trim_silence,augment
from .training import load_dataset
from .native_evaluation import _examples


def prepare_development(base,output):
    base=Path(base).absolute();output=Path(output).absolute()
    if base.resolve()!=base or output.resolve()!=output or output.exists():
        raise ValueError('real development input and fresh output required')
    raw=(base/'manifest.json').read_bytes();manifest=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if manifest.get('role')!='development':raise ValueError('explicit development role required')
    grouped={label:[] for label in range(6)};targets={}
    for row in manifest['records']:
        if 'sha256' in row:
            if row['label'] not in grouped:raise ValueError('spoken development label required')
            grouped[row['label']].append(row['id'])
            file=base/'sources'/(row['id']+'.wav')
            if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('development source changed')
    for label,ids in grouped.items():
        if len(ids)<3:raise ValueError('three source utterances per spoken class required')
        ids=sorted(ids);n=max(1,len(ids)//10)
        for index,ident in enumerate(ids):targets[ident]='validation' if index<n else 'test' if index<2*n else 'train'
    data=load_dataset(base)
    for _ in _examples(base):pass
    arrays={s:dict(x=[],y=[]) for s in ('train','validation','test')};positions={s:0 for s in arrays};records=[];copies=[]
    for original in manifest['records']:
        row=dict(original);old_split=row['split'];count=manifest['variants'] if 'sha256' in row else 1
        offset=positions[old_split];positions[old_split]+=count
        if 'sha256' in row:
            target=targets[row['id']]
            for nonce in range(10000):
                source=f"dev-balanced-v1/{digest}/{row['id']}/{nonce}"
                if split_for(source)==target:break
            else:raise ValueError('bounded split identity search exhausted')
            ident=hashlib.sha256(source.encode()).hexdigest();file=base/'sources'/(row['id']+'.wav')
            pcm=trim_silence(read_pcm(file));copies.append((file,ident))
            row.update(origin_id=row['id'],origin_source=row['source'],origin_split=old_split,source=source,id=ident,split=target)
            values=[features(augment(pcm,int(ident[:8],16)+i)) for i in range(count)]
        else:values=data[old_split]['x'][offset:offset+count]
        out=arrays[row['split']];out['x'].extend(values);out['y'].extend([row['label']]*count);records.append(row)
    if any(set(out['y'])!=set(range(7)) for out in arrays.values()):
        raise ValueError('all seven classes required in every development split')
    output.mkdir(parents=True);(output/'sources').mkdir()
    for file,ident in copies:shutil.copyfile(file,output/'sources'/(ident+'.wav'))
    for split,out in arrays.items():np.savez_compressed(output/(split+'.npz'),x=np.asarray(out['x'],dtype=np.float32),y=np.asarray(out['y'],dtype=np.int64))
    manifest.update(records=records,origin_manifest_sha256=digest,development_split='source-stratified-before-training-v1')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(class_coverage_complete=True,origin_manifest_sha256=digest,
                source_counts={s:sum(r['split']==s for r in records) for s in arrays},final_conditions_used=False,
                real_voice_tested=False,device_tested=False)
