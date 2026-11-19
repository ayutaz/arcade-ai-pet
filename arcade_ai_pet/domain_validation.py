"""Validation gates per development domain; never read a final-role corpus."""
import hashlib,json,re
from pathlib import Path
from .channel_training import make_arrays
from .channel_freeze import _rows,_accepted
from .frozen import read_layers
from .training import scores


def validate_domains(checkpoint,domains):
    checkpoint=Path(checkpoint);meta=json.loads((checkpoint/'validation.json').read_text())
    if (hashlib.sha256((checkpoint/'weights.npz').read_bytes()).hexdigest()!=meta['weights_sha256'] or
            type(meta.get('threshold_q10')) is not int or not 0<=meta['threshold_q10']<=4096 or
            meta.get('candidate')!='hp160' or meta.get('frontend_version')!='adaptive' or
            type(domains) is not dict or not 2<=len(domains)<=4 or
            any(type(k) is not str or not re.fullmatch('[a-z][a-z0-9-]{0,31}',k) for k in domains)):
        raise ValueError('fixed checkpoint and bounded development domains required')
    seen=set();waves=set();paths={name:Path(path) for name,path in domains.items()};manifests={}
    for name,path in paths.items():
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('role','development')!='development':raise ValueError('final domain forbidden')
        for row in manifest['records']:
            if row['id'] in seen or 'sha256' in row and row['sha256'] in waves:
                raise ValueError('development domains overlap')
            seen.add(row['id'])
            if 'sha256' in row:waves.add(row['sha256'])
        manifests[name]=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    layers,_=read_layers(checkpoint/'weights.npz');reports={}
    for name,path in paths.items():
        validation=make_arrays(path,meta['candidate'],frontend=meta['frontend_version'])['validation']
        rows=_rows(validation['y'],scores(validation['x'],layers),validation['profiles'],meta['candidate'],meta['threshold_q10'])
        reports[name]=dict(profiles=rows,accepted=_accepted(rows),manifest_sha256=manifests[name])
    return dict(domains=reports,accepted_all_domains=all(r['accepted'] for r in reports.values()),
                weights_sha256=meta['weights_sha256'],threshold_q10=meta['threshold_q10'],
                validation_only=True,real_voice_tested=False,device_tested=False)
