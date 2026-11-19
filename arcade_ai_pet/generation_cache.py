"""Reuse hash-verified source WAVs; never reuse ASR decisions or overwrite prior output."""
import hashlib
import json
import os
from pathlib import Path
import re
from .data import SOURCES
from .neural_corpus import validate_plan,_digest
from .generated_audio import _read_wave


def source_callback(plan,prior,*,synthesize):
    digest=validate_plan(plan)
    plan=json.loads(json.dumps(plan))
    prior=Path(prior).absolute()
    if (prior.resolve()!=prior or not callable(synthesize) or
            any((prior/name).exists() for name in ('manifest.json','train.npz','validation.npz','test.npz'))):
        raise ValueError('real unpartitioned source cache and teacher required')
    if validate_plan(json.loads((prior/'plan.json').read_text()))!=digest:
        raise ValueError('source cache plan differs')
    progress=json.loads((prior/'progress.json').read_text())
    if progress.get('plan_sha256')!=digest or progress.get('status') not in ('incomplete','qualified'):
        raise ValueError('bounded source cache progress required')
    expected={}
    for condition in plan['conditions']:
        if condition['role']!='development':
            continue
        for label,texts in SOURCES.items():
            for prompt in texts:
                for attempt in range(plan['attempts']):
                    source=f"neural/{digest}/{condition['id']}/{label}/{_digest(prompt)}/{attempt}"
                    ident=hashlib.sha256(source.encode()).hexdigest()
                    seed=int.from_bytes(hashlib.sha256(f"{plan['seed']}/{source}".encode()).digest()[:4],'little')
                    expected[ident]=dict(prompt=prompt,caption=condition['caption'],seed=seed,label=label,condition_id=condition['id'])
    cached={}
    for row in progress['attempts']:
        wave_hash=row.get('wave_sha256')
        if type(wave_hash) is not str or not re.fullmatch('[0-9a-f]{64}',wave_hash):
            continue
        item=expected.get(row.get('id'))
        if item is None or row['id'] in cached or any(row.get(key)!=item[key] for key in ('prompt','label','condition_id','seed')):
            raise ValueError('cached source identity differs or uses final conditions')
        cached[row['id']]=wave_hash
    stats=dict(reused=0,synthesized=0)
    def generate(prompt,caption,seed,path):
        path=Path(path).absolute()
        item=expected.get(path.stem)
        if (path.resolve()!=path or path.exists() or path.name!=path.stem+'.wav' or item is None or
                type(prompt) is not str or type(caption) is not str or type(seed) is not int or
                (prompt,caption,seed)!=(item['prompt'],item['caption'],item['seed'])):
            raise ValueError('fresh output and exact development source identity required')
        if path.stem in cached:
            raw,_,_=_read_wave(prior/'sources'/path.name)
            if hashlib.sha256(raw).hexdigest()!=cached[path.stem]:
                raise ValueError('cached source WAV changed; do not silently regenerate')
            with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as stream:
                stream.write(raw)
            stats['reused']+=1
        else:
            synthesize(prompt,caption,seed,path)
            stats['synthesized']+=1
    return generate,stats
