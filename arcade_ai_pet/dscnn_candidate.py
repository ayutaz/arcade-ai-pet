"""Bounded host DS-CNN candidate with the unchanged development caption holdout."""
import json
import math
from pathlib import Path
import numpy as np
from .decision_assets import real_path, digest
from .caption_training import _domains, _input_hashes, _quality, SOURCES as SHARED_SOURCES
from .caption_holdout import select_holdout, make_arrays as load_neural_arrays
from .logmel_arrays import make_arrays as load_legacy_arrays
from .dscnn_training import fit
from .dscnn_model import scores, weights_bytes
from .class_rejection import calibrate, decisions
from .channel_robustness import PROFILES, measure

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (*SHARED_SOURCES, 'arcade_ai_pet/logmel_frontend.py', 'arcade_ai_pet/logmel_arrays.py',
           'arcade_ai_pet/dscnn_model.py', 'arcade_ai_pet/dscnn_quantize.py',
           'arcade_ai_pet/dscnn_training.py', 'arcade_ai_pet/dscnn_candidate.py',
           'arcade_ai_pet/envelope_frontend.py')


def _source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def train_candidate(domains, output, *, holdout_seed, epochs=30, batch_size=256, neural_weight=2,
                    max_passes=4, device='cpu', max_steps=8000, max_seconds=900, frontend='context-logmel64'):
    output = real_path(output)
    if (type(frontend) is not str or frontend not in ('context-logmel64','context-logmel64-envelope') or
        output.exists() or type(epochs) is not int or not 1 <= epochs <= 1000 or
        type(batch_size) is not int or not 1 <= batch_size <= 1024 or
        type(neural_weight) is not int or not 1 <= neural_weight <= 8 or
        type(holdout_seed) is not int or not 0 <= holdout_seed < 2**31 or
        type(max_passes) is not int or not 1 <= max_passes <= 8 or
        device not in ('cpu', 'mps') or type(max_steps) is not int or not 1 <= max_steps <= 1000000 or
        type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 86400):
        raise ValueError('fresh output and bounded local DS-CNN recipe required')
    mode = dict(envelope=True) if frontend == 'context-logmel64-envelope' else {}
    paths = _domains(domains)
    before, code = _input_hashes(paths), _source_hashes()
    selection = select_holdout(paths['neural'], seed=holdout_seed)
    options = dict(noise_replicas=5, hum_replicas=3, noise_mode='fresh')
    legacy = load_legacy_arrays(paths['legacy'], 'hp160', frontend=frontend, **options)
    neural = load_neural_arrays(paths['neural'], selection=selection, frontend=frontend, **options)
    if set(legacy) != {'train', 'validation'} or set(neural) != {'train', 'validation', 'held_validation'}:
        raise ValueError('only explicit development arrays allowed')
    ids = [set(neural[name]['source_ids']) for name in ('train', 'validation', 'held_validation')]
    if any(ids[a] & ids[b] for a in range(3) for b in range(a+1, 3)):
        raise ValueError('development source traces overlap')
    train_rows = len(legacy['train']['y']) + neural_weight * len(neural['train']['y'])
    if train_rows > 65536:
        raise ValueError('host training matrix exceeds the row budget')
    primary = [legacy['validation'], neural['held_validation']]
    x = np.concatenate((legacy['train']['x'], np.tile(neural['train']['x'], (neural_weight, 1))))
    y = np.concatenate((legacy['train']['y'], np.tile(neural['train']['y'], neural_weight)))
    vx, vy = (np.concatenate([row[key] for row in primary]) for key in ('x', 'y'))
    groups = np.asarray([domain * 8 + PROFILES.index(profile) for domain, row in enumerate(primary) for profile in row['profiles']], dtype=np.int64)
    result = fit(x, y, vx, vy, groups, epochs=epochs, batch_size=batch_size, seed=42,
                 device=device, max_steps=max_steps, max_seconds=max_seconds, **mode)
    layers = result.pop('layers')
    calibrated = calibrate(vy, scores(vx, layers, **mode), groups,
                           initial_threshold=result['checkpoint_threshold_q10'], max_passes=max_passes)
    table = np.asarray(calibrated['thresholds_q10'], dtype=np.int32)
    familiar = neural['validation']
    fp = np.asarray([PROFILES.index(p) for p in familiar['profiles']])
    familiar_predictions = decisions(scores(familiar['x'], layers, **mode), table)
    familiar_rows = [dict(domain='familiar-neural', profile=profile,
                          **measure(familiar['y'][fp == i], familiar_predictions[fp == i])) for i, profile in enumerate(PROFILES)]
    rows = [dict(row, domain='legacy' if row['domain'] == 'legacy' else 'held-neural')
            for row in calibrated['domain_profiles']] + familiar_rows
    all_y = np.concatenate((vy, familiar['y']))
    all_predictions = np.concatenate((decisions(scores(vx, layers, **mode), table), familiar_predictions))
    all_profiles = np.concatenate((groups % 8, fp))
    pooled = [dict(profile=profile, **measure(all_y[all_profiles == i], all_predictions[all_profiles == i])) for i, profile in enumerate(PROFILES)]
    if before != _input_hashes(paths) or code != _source_hashes():
        raise ValueError('development inputs or source changed during fit')
    size = weights_bytes(layers, **mode)
    if size + 20 > 24576:
        raise ValueError('24KiB numeric model budget')
    result.update(scope='development_logmel_dscnn_not_final', classifier='logmel-dscnn-envelope-v1' if mode else 'logmel-dscnn-v1', frontend=frontend,
                  thresholds_q10=calibrated['thresholds_q10'], domain_profiles=rows, pooled_profiles=pooled,
                  selection_domains=['legacy_validation', 'held_neural_validation'],
                  additional_required_regression=['original_neural_validation'],
                  selection_profiles=calibrated['domain_profiles'], selection_pooled_profiles=calibrated['pooled_profiles'],
                  selection_accepted=calibrated['accepted_validation'],
                  accepted_validation=bool(calibrated['accepted_validation'] and _quality(rows) and _quality(pooled)),
                  calibration_evaluations=calibrated['evaluations'], calibration_passes=calibrated['passes_per_start'],
                  selection=selection, source_sha256=code, original_input_sha256=before,
                  original_archives_unchanged=True, held_sources_absent_from_train=True,
                  split_examples=dict(train=len(y), selection_validation=len(vy), familiar_validation=len(familiar['y'])),
                  neural_weight=neural_weight, noise_replicas=5, hum_replicas=3, noise_mode='fresh',
                  weights_bytes=size, numeric_bytes=size+20, frozen=False, final_evaluated=False,
                  device_tested=False, real_voice_tested=False, vast_usd=0)
    output.mkdir(parents=True, mode=0o700)
    np.savez(output / 'weights.npz', **{f'{field}{i}': layer[field] for i, layer in enumerate(layers) for field in ('weight', 'bias', 'shift')})
    result['weights_sha256'] = digest(output / 'weights.npz')
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
