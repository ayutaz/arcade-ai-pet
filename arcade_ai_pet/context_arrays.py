"""Context-front-end arrays from existing development WAVs; no data generation."""
import hashlib,json
from pathlib import Path
import numpy as np
from .channel_robustness import PROFILES,condition,simulate
from .channel_training import level,_noise_options
from .native_evaluation import _examples
from .context_frontend import features


def make_arrays(dataset,candidate,*,frontend='context',noise_replicas=1,noise_mode='fresh',hum_replicas=1):
    if candidate!='hp160' or frontend!='context':raise ValueError('explicit hp160/context experiment required')
    _noise_options(noise_replicas,noise_mode,hum_replicas)
    dataset=Path(dataset);manifest=json.loads((dataset/'manifest.json').read_text())
    expanded=[(row['split'],row['id']) for row in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in row else 1)]
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
                out['x'].append(features(level(condition(simulate(pcm,profile,current),candidate))))
                out['y'].append(label);out['profiles'].append(profile)
    for out in arrays.values():
        if not out['y']:raise ValueError('nonempty train and validation required')
        out['x']=np.asarray(out['x'],dtype=np.float32);out['y']=np.asarray(out['y'],dtype=np.int64)
    return arrays
