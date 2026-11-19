"""Development-only caption holdout without editing source archives or test roles."""
import hashlib
import json
import re
import numpy as np
from .decision_assets import real_path, digest
from .features import split_for
from .native_evaluation import _examples
from .channel_robustness import PROFILES, simulate, condition
from .channel_training import level, _noise_options
from .cmn_frontend import features
from . import logmel_frontend, envelope_frontend


def _manifest(dataset):
    dataset = real_path(dataset)
    path = real_path(dataset / 'manifest.json')
    raw = path.read_bytes()
    meta = json.loads(raw)
    if (meta.get('role') != 'development' or meta.get('feature_version') != 'aligned-mfcc-v2' or
        type(meta.get('variants')) is not int or not 1 <= meta['variants'] <= 10 or
        type(meta.get('records')) is not list or not 1 <= len(meta['records']) <= 10000):
        raise ValueError('bounded development source manifest required')
    ids, waves = set(), set()
    for row in meta['records']:
        source, ident, label = row.get('source'), row.get('id'), row.get('label')
        if (type(source) is not str or type(ident) is not str or
            hashlib.sha256(source.encode()).hexdigest() != ident or ident in ids or
            type(label) is not int or not 0 <= label <= 6 or
            row.get('split') not in ('train', 'validation', 'test') or split_for(source) != row['split']):
            raise ValueError('unique original source/split/label identity required')
        ids.add(ident)
        if 'sha256' in row:
            wave = row['sha256']; caption = row.get('condition_id')
            if (type(wave) is not str or not re.fullmatch('[0-9a-f]{64}', wave) or wave in waves or
                type(caption) is not str or not re.fullmatch('[a-z][a-z0-9-]{0,31}', caption) or label == 6):
                raise ValueError('unique caption speech identity required')
            waves.add(wave)
            real_path(dataset / 'sources' / (ident + '.wav'))
        elif label != 6 or 'condition_id' in row:
            raise ValueError('separate silence control identity required')
    return dataset, meta, hashlib.sha256(raw).hexdigest()


def select_holdout(dataset, *, seed):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('bounded integer holdout seed required')
    _, meta, fingerprint = _manifest(dataset)
    splits = {}
    for row in meta['records']:
        if 'condition_id' in row:
            splits.setdefault(row['condition_id'], set()).add(row['split'])
    captions = [name for name, roles in splits.items() if roles == {'train'}]
    controls = [row['id'] for row in meta['records'] if row['label'] == 6 and row['split'] == 'train']
    if len(captions) < 5 or len(controls) < 20:
        raise ValueError('four held captions, remaining train caption and twenty controls required')
    captions.sort(key=lambda value: hashlib.sha256(f'{seed}/condition/{value}'.encode()).hexdigest())
    controls.sort(key=lambda value: hashlib.sha256(f'{seed}/control/{value}'.encode()).hexdigest())
    return dict(seed=seed, manifest_sha256=fingerprint, held_condition_ids=captions[:4], held_control_ids=controls[:20])


def _checksums(dataset):
    return {name: digest(real_path(dataset / name)) for name in ('manifest.json', 'train.npz', 'validation.npz', 'test.npz')}


def make_arrays(dataset, *, selection, noise_replicas=5, noise_mode='fresh', hum_replicas=3,
                frontend='context-duration-cmn'):
    if frontend not in ('context-duration-cmn', 'context-logmel64', 'context-logmel64-envelope'):
        raise ValueError('explicit supported development frontend required')
    feature_fn = {'context-duration-cmn': features, 'context-logmel64': logmel_frontend.features,
                  'context-logmel64-envelope': envelope_frontend.features}[frontend]
    _noise_options(noise_replicas, noise_mode, hum_replicas)
    if type(selection) is not dict:
        raise ValueError('explicit deterministic development selection required')
    dataset, manifest, fingerprint = _manifest(dataset)
    if fingerprint != selection.get('manifest_sha256'):
        raise ValueError('selection does not bind the parsed manifest')
    if select_holdout(dataset, seed=selection.get('seed')) != selection:
        raise ValueError('holdout selection or original manifest changed')
    before = _checksums(dataset)
    if before['manifest.json'] != fingerprint:
        raise ValueError('manifest changed before reconstruction')
    held_captions, held_controls = set(selection['held_condition_ids']), set(selection['held_control_ids'])
    expanded = [(row, variant) for row in manifest['records']
                for variant in range(manifest['variants'] if 'sha256' in row else 1)]
    arrays = {name: dict(x=[], y=[], profiles=[], source_ids=[], variants=[], condition_ids=[])
              for name in ('train', 'validation', 'held_validation')}
    for (pcm, label, _), (row, variant) in zip(_examples(dataset), expanded, strict=True):
        if label != row['label']:
            raise ValueError('reconstructed source label changed')
        if row['split'] == 'test':
            continue
        ident = row['id']
        held = row.get('condition_id') in held_captions or ident in held_controls
        split = 'validation' if row['split'] == 'validation' else 'held_validation' if held else 'train'
        out = arrays[split]
        seed = int(ident[:16], 16)
        for profile in PROFILES:
            replicas = (noise_replicas if profile == 'noise' else hum_replicas if profile == 'hum' else 1) if split == 'train' else 1
            for replica in range(replicas):
                current = seed
                if replica and profile == 'noise' and noise_mode == 'fresh':
                    current = int.from_bytes(hashlib.sha256(ident.encode() + pcm.tobytes() + bytes([replica])).digest()[:8], 'little')
                out['x'].append(feature_fn(level(condition(simulate(pcm, profile, current), 'hp160'))))
                out['y'].append(label); out['profiles'].append(profile)
                out['source_ids'].append(ident); out['variants'].append(variant)
                out['condition_ids'].append(row.get('condition_id', ''))
    for out in arrays.values():
        if not out['y']:
            raise ValueError('nonempty training, familiar validation and held validation required')
        out['x'] = np.asarray(out['x'], dtype=np.float32)
        out['y'] = np.asarray(out['y'], dtype=np.int64)
    memberships = [set(out['source_ids']) for out in arrays.values()]
    if any(memberships[a] & memberships[b] for a in range(3) for b in range(a+1, 3)):
        raise ValueError('development source membership overlap')
    if before != _checksums(dataset):
        raise ValueError('original manifest or split archives changed during reconstruction')
    return arrays
