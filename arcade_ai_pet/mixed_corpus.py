"""Merge development sources with original splits and source-level provenance."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from .training import load_dataset
from .native_evaluation import _examples


def merge_development(inputs,output):
    inputs=[Path(p).absolute() for p in inputs];output=Path(output).absolute()
    if (not 1<=len(inputs)<=8 or output.exists() or output.resolve()!=output or
            any(p.resolve()!=p or not p.is_dir() for p in inputs)):
        raise ValueError('bounded real development inputs and fresh output required')
    components=[];records=[];datasets=[];ids=set();speech_hashes=set();variants=None
    for path in inputs:
        raw=(path/'manifest.json').read_bytes();manifest=json.loads(raw)
        if (manifest.get('role','development')!='development' or manifest.get('feature_version')!='aligned-mfcc-v2' or
                not isinstance(manifest.get('license'),str) or not manifest['license'] or
                variants is not None and manifest.get('variants')!=variants):
            raise ValueError('compatible licensed development inputs required')
        variants=manifest['variants']
        for row in manifest['records']:
            if row['id'] in ids:raise ValueError('duplicate source identity')
            ids.add(row['id'])
            if 'sha256' in row:
                file=path/'sources'/(row['id']+'.wav')
                if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('source content changed')
                if row['sha256'] in speech_hashes:raise ValueError('duplicate speech content')
                speech_hashes.add(row['sha256'])
        data=load_dataset(path)
        # Validate saved features against the same WAVs/noise before any output.
        for _ in _examples(path):pass
        components.append(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),
                               **{k:v for k,v in manifest.items() if k!='records'}))
        records.extend(manifest['records']);datasets.append(data)
    output.mkdir(parents=True);(output/'sources').mkdir()
    for path in inputs:
        manifest=json.loads((path/'manifest.json').read_text())
        for row in manifest['records']:
            if 'sha256' in row:shutil.copyfile(path/'sources'/(row['id']+'.wav'),output/'sources'/(row['id']+'.wav'))
    for split in ('train','validation','test'):
        np.savez_compressed(output/(split+'.npz'),x=np.concatenate([d[split]['x'] for d in datasets]),y=np.concatenate([d[split]['y'] for d in datasets]))
    result=dict(schema=1,role='development',variants=variants,feature_version='aligned-mfcc-v2',
                engine='source-preserving development mixture',license='component licenses apply; see components',
                components=components,records=records)
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(speech_sources=len(speech_hashes),source_records=len(records),real_voice_tested=False)
