"""Qualified development corpus; final-condition synthesis is deliberately separate."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
from .data import SOURCES, read_pcm, trim_silence, augment
from .features import features, split_for
from .generated_audio import audit_generated_wave


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_plan(plan):
    if (type(plan) is not dict or set(plan) != {'schema','seed','variants','attempts','provider','conditions'} or
            type(plan.get('schema')) is not int or plan['schema'] != 1 or
            type(plan.get('seed')) is not int or not 0 <= plan['seed'] < 2**31 or
            type(plan.get('variants')) is not int or not 1 <= plan['variants'] <= 10 or
            type(plan.get('attempts')) is not int or not 1 <= plan['attempts'] <= 3):
        raise ValueError('bounded source plan required')
    provider=plan['provider']; hashes=('model_sha256','codec_sha256','runtime_lock_sha256','asr_model_sha256')
    if (type(provider) is not dict or set(provider) != {'name','license','engine_commit',*hashes} or
            provider['name'] != 'irodori-v2-voice-design' or provider['license'] != 'MIT' or
            not isinstance(provider['engine_commit'],str) or not re.fullmatch('[0-9a-f]{40}',provider['engine_commit']) or
            any(type(provider[k]) is not str or not re.fullmatch('[0-9a-f]{64}',provider[k]) for k in hashes)):
        raise ValueError('reviewed provider identity required')
    conditions=plan['conditions'];ids=set();captions=set();roles=set()
    if type(conditions) is not list or not 2 <= len(conditions) <= 32:
        raise ValueError('development and final conditions required')
    for row in conditions:
        if (type(row) is not dict or set(row) != {'id','role','caption'} or
                type(row['id']) is not str or not re.fullmatch('[a-z][a-z0-9-]{0,31}',row['id']) or
                row['id'] in ids or row['role'] not in ('development','final') or
                type(row['caption']) is not str or not 1 <= len(row['caption']) <= 240):
            raise ValueError('unique bounded condition required')
        caption=unicodedata.normalize('NFKC',row['caption']).strip()
        if not caption or caption in captions:
            raise ValueError('caption conditions must be disjoint')
        ids.add(row['id']);captions.add(caption);roles.add(row['role'])
    if roles != {'development','final'}:
        raise ValueError('both condition roles required')
    return _digest(plan)


def generate_development(plan, output, *, synthesize, recognize, finalize=True):
    return _generate(plan,output,synthesize=synthesize,recognize=recognize,finalize=finalize,role='development')


def _generate(plan,output,*,synthesize,recognize,finalize,role,model_binding=None):
    digest=validate_plan(plan);output=Path(output).absolute()
    if not callable(synthesize) or not callable(recognize) or output.resolve()!=output or output.exists():
        raise ValueError('fresh real output and provider callbacks required')
    fingerprints=dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      audit_sha256=hashlib.sha256(Path(__file__).with_name('generated_audio.py').read_bytes()).hexdigest())
    plan=json.loads(json.dumps(plan));output.mkdir(parents=True);(output/'sources').mkdir()
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    progress=dict(status='incomplete',plan_sha256=digest,attempts=[])
    def save(): (output/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    save();accepted=[];seen=set();covered=set()
    for condition in plan['conditions']:
        if condition['role'] != role:continue
        for label,texts in SOURCES.items():
            for prompt in texts:
                for attempt in range(plan['attempts']):
                    source=f"neural/{digest}/{condition['id']}/{label}/{_digest(prompt)}/{attempt}"
                    ident=hashlib.sha256(source.encode()).hexdigest()
                    seed=int.from_bytes(hashlib.sha256(f"{plan['seed']}/{source}".encode()).digest()[:4],'little')
                    file=output/'sources'/(ident+'.wav')
                    try:
                        synthesize(prompt,condition['caption'],seed,file)
                    except Exception:
                        progress['attempts'].append(dict(id=ident,label=label,status='provider_error',content_check_passed=False));save()
                        raise RuntimeError('source provider stopped; preserve partial output') from None
                    try: audit=audit_generated_wave(file,prompt=prompt,recognize=recognize)
                    except ValueError: audit=dict(label=label,status='invalid_source',content_check_passed=False)
                    audit.update(id=ident,condition_id=condition['id'],seed=seed)
                    if audit.get('wave_sha256') in seen:
                        audit.update(status='duplicate_source',content_check_passed=False)
                    progress['attempts'].append(audit);save()
                    if not audit['content_check_passed']:continue
                    seen.add(audit['wave_sha256']);covered.add(prompt)
                    accepted.append(dict(source=source,id=ident,split=split_for(source),label=label,
                                         sha256=audit['wave_sha256'],condition_id=condition['id'],prompt=prompt))
                    break
    if covered != {text for texts in SOURCES.values() for text in texts}:
        raise ValueError('incomplete qualified prompt coverage; preserve partial output')
    qualified=dict(schema=1,role=role,plan_sha256=digest,records=accepted,
                   **fingerprints)
    if model_binding is not None:qualified['model_binding']=model_binding
    (output/'qualified-sources.json').write_text(json.dumps(qualified,ensure_ascii=False,indent=2)+'\n')
    progress['status']='qualified';save()
    if finalize:
        return _finalize(output,role=role)
    return dict(qualified_speech_sources=len(accepted),plan_sha256=digest,
                real_voice_tested=False,model_trained=False,final_conditions_generated=(role=='final'))


def finalize_development(output):
    """Build core-version features from frozen qualified WAVs without invoking models."""
    return _finalize(output,role='development')


def _finalize(output,*,role):
    output=Path(output).absolute()
    if output.resolve()!=output or any((output/name).exists() for name in ('manifest.json','train.npz','validation.npz','test.npz')):
        raise ValueError('fresh real feature destination required')
    plan=json.loads((output/'plan.json').read_text());digest=validate_plan(plan)
    qualified=json.loads((output/'qualified-sources.json').read_text())
    accepted=qualified['records'];conditions={r['id']:r for r in plan['conditions'] if r['role']==role}
    if qualified.get('schema')!=1 or qualified.get('role')!=role or qualified.get('plan_sha256')!=digest:
        raise ValueError('qualified plan identity changed')
    progress=json.loads((output/'progress.json').read_text())
    if progress.get('status')!='qualified' or progress.get('plan_sha256')!=digest:
        raise ValueError('completed qualification required')
    audits={r['id']:r for r in progress['attempts'] if r.get('content_check_passed') is True}
    covered=set();ids=set();hashes=set()
    for row in accepted:
        condition=conditions.get(row['condition_id']);source=row['source'];audit=audits.get(row['id'],{})
        if any(audit.get(k)!=row[v] for k,v in (('wave_sha256','sha256'),('label','label'),('prompt','prompt'),('condition_id','condition_id'))):
            raise ValueError('qualified audit changed')
        prefix=f"neural/{digest}/{row['condition_id']}/{row['label']}/{_digest(row['prompt'])}/"
        if (condition is None or row['prompt'] not in SOURCES.get(row['label'],()) or
                not source.startswith(prefix) or source[len(prefix):] not in map(str,range(plan['attempts'])) or
                row['id']!=hashlib.sha256(source.encode()).hexdigest() or row['id'] in ids or
                row['split']!=split_for(source) or row['sha256'] in hashes):
            raise ValueError('qualified source identity changed')
        path=output/'sources'/(row['id']+'.wav')
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('qualified source changed')
        covered.add(row['prompt']);ids.add(row['id']);hashes.add(row['sha256'])
    if covered!={t for texts in SOURCES.values() for t in texts}:
        raise ValueError('incomplete qualified prompt coverage')
    arrays={split:dict(x=[],y=[]) for split in ('train','validation','test')}
    records=list(accepted)
    for row in accepted:
        path=output/'sources'/(row['id']+'.wav')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('qualified source changed')
        pcm=trim_silence(read_pcm(path))
        for variant in range(plan['variants']):
            out=arrays[row['split']];out['x'].append(features(augment(pcm,int(row['id'][:8],16)+variant)));out['y'].append(row['label'])
    namespace=('neural-dev-' if role=='development' else 'neural-final-')+digest[:12]
    for index in range(200):
        source=f'{namespace}/{index}';ident=hashlib.sha256(source.encode()).hexdigest();split=split_for(source)
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(source.encode()).digest()[:8],'little'))
        pcm=np.clip(rng.normal(0,0 if index%4==0 else rng.uniform(1,200),32000),-32767,32767).astype(np.int16)
        arrays[split]['x'].append(features(pcm));arrays[split]['y'].append(6)
        records.append(dict(source=source,id=ident,split=split,label=6,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    for split,out in arrays.items():
        np.savez_compressed(output/(split+'.npz'),x=np.asarray(out['x'],dtype=np.float32).reshape(-1,260),y=np.asarray(out['y'],dtype=np.int64))
    manifest=dict(schema=1,engine='Irodori VoiceDesign generated speech with independent ASR content checks',
                  role=role,plan_sha256=digest,provider=plan['provider'],variants=plan['variants'],
                  generator_sha256=qualified['generator_sha256'],
                  audit_sha256=qualified['audit_sha256'],feature_numpy_version=np.__version__,
                  license='declared MIT provider; see provider artifacts and attribution',
                  feature_version='aligned-mfcc-v2',records=records)
    if 'model_binding' in qualified:manifest['model_binding']=qualified['model_binding']
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    progress=json.loads((output/'progress.json').read_text());progress['status']='complete'
    (output/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    return dict(qualified_speech_sources=len(accepted),records=len(records),plan_sha256=digest,
                real_voice_tested=False,model_trained=False,final_conditions_generated=(role=='final'))
