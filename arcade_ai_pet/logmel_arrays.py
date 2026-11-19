"""Development-only log-mel arrays with unchanged original source/profile identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .channel_robustness import PROFILES, condition, simulate
from .channel_training import level, _noise_options
from .native_evaluation import _examples
from .logmel_frontend import features


def make_arrays(dataset, candidate, *, frontend='context-logmel64', noise_replicas=1,
                noise_mode='fresh', hum_replicas=1):
    if candidate != 'hp160' or frontend != 'context-logmel64':
        raise ValueError('explicit hp160/context-logmel64 experiment required')
    _noise_options(noise_replicas, noise_mode, hum_replicas)
    dataset = Path(dataset)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('role', 'development') != 'development':
        raise ValueError('final logmel corpus forbidden')
    expanded = [row for row in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in row else 1)]
    arrays = {split: dict(x=[], y=[], profiles=[]) for split in ('train', 'validation')}
    for (pcm, label, _), row in zip(_examples(dataset), expanded, strict=True):
        if label != row['label']:
            raise ValueError('reconstructed source label changed')
        if row['split'] not in arrays:
            continue
        out = arrays[row['split']]
        seed = int(row['id'][:16], 16)
        for profile in PROFILES:
            replicas = (noise_replicas if profile == 'noise' else hum_replicas if profile == 'hum' else 1) if row['split'] == 'train' else 1
            for replica in range(replicas):
                current = seed
                if replica and profile == 'noise' and noise_mode == 'fresh':
                    current = int.from_bytes(hashlib.sha256(row['id'].encode() + pcm.tobytes() + bytes([replica])).digest()[:8], 'little')
                out['x'].append(features(level(condition(simulate(pcm, profile, current), candidate))))
                out['y'].append(label); out['profiles'].append(profile)
    for out in arrays.values():
        if not out['y']:
            raise ValueError('nonempty train and validation required')
        out['x'] = np.asarray(out['x'], dtype=np.float32)
        out['y'] = np.asarray(out['y'], dtype=np.int64)
    return arrays
