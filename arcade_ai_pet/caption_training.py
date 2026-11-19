"""Select on held development captions; require familiar regression without fitting final data."""
import json
from pathlib import Path
import re
import numpy as np
from .decision_assets import real_path, digest
from .channel_freeze import SOURCES as LEGACY_SOURCES
from .cmn_arrays import make_arrays as load_legacy_arrays
from .caption_holdout import select_holdout, make_arrays as load_neural_arrays
from .domain_training import fit
from .class_rejection import calibrate, decisions
from .channel_robustness import PROFILES, measure
from .training import scores

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(dict.fromkeys((*LEGACY_SOURCES, 'arcade_ai_pet/caption_training.py',
    'arcade_ai_pet/caption_holdout.py', 'arcade_ai_pet/domain_training.py',
    'arcade_ai_pet/class_rejection.py', 'arcade_ai_pet/cmn_arrays.py',
    'arcade_ai_pet/cmn_frontend.py', 'arcade_ai_pet/duration_arrays.py',
    'arcade_ai_pet/duration_frontend.py', 'arcade_ai_pet/duration_quantize.py',
    'arcade_ai_pet/context_frontend.py')))


def _quality(rows):
    return bool(rows) and all(r['macro_f1'] >= .90 and r['false_activation_rate'] <= .05 and r['command_recall'] >= .80 for r in rows)


def _domains(domains):
    if type(domains) is not dict or set(domains) != {'legacy', 'neural'}:
        raise ValueError('two original development domains required')
    paths = {name: real_path(domains[name]) for name in ('legacy', 'neural')}
    seen, waves = set(), set()
    for name, path in paths.items():
        manifest = json.loads(real_path(path / 'manifest.json').read_text())
        if manifest.get('role', 'development') != 'development' or not manifest.get('records'):
            raise ValueError('nonempty development domains required')
        for row in manifest['records']:
            ident, wave = row.get('id'), row.get('sha256')
            if (type(ident) is not str or not re.fullmatch('[0-9a-f]{64}', ident) or ident in seen or
                wave is not None and (type(wave) is not str or not re.fullmatch('[0-9a-f]{64}', wave) or wave in waves)):
                raise ValueError('development source/wave overlap or invalid identity')
            seen.add(ident)
            if wave is not None:
                waves.add(wave)
    return paths


def _input_hashes(paths):
    return {name + '/' + file: digest(real_path(path / file)) for name, path in paths.items()
            for file in ('manifest.json', 'train.npz', 'validation.npz', 'test.npz')}


def _source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def train_candidate(domains, output, *, holdout_seed, epochs=1000, hidden=88, neural_weight=2, max_passes=4):
    output = real_path(output)
    if (output.exists() or type(epochs) is not int or not 1 <= epochs <= 1000 or
        type(hidden) is not int or not 1 <= hidden <= 90 or
        type(neural_weight) is not int or not 1 <= neural_weight <= 8 or
        type(max_passes) is not int or not 1 <= max_passes <= 8 or
        type(holdout_seed) is not int or not 0 <= holdout_seed < 2**31):
        raise ValueError('fresh output and bounded caption training recipe required')
    paths = _domains(domains)
    before, code = _input_hashes(paths), _source_hashes()
    selection = select_holdout(paths['neural'], seed=holdout_seed)
    options = dict(noise_replicas=5, hum_replicas=3, noise_mode='fresh')
    legacy = load_legacy_arrays(paths['legacy'], 'hp160', **options)
    neural = load_neural_arrays(paths['neural'], selection=selection, **options)
    if set(legacy) != {'train', 'validation'} or set(neural) != {'train', 'validation', 'held_validation'}:
        raise ValueError('only explicit development arrays allowed; test cannot be forwarded')
    ids = [set(neural[name]['source_ids']) for name in ('train', 'validation', 'held_validation')]
    if any(ids[a] & ids[b] for a in range(3) for b in range(a+1, 3)):
        raise ValueError('held and familiar development source traces overlap')
    primary = [legacy['validation'], neural['held_validation']]
    x = np.concatenate((legacy['train']['x'], np.tile(neural['train']['x'], (neural_weight, 1))))
    y = np.concatenate((legacy['train']['y'], np.tile(neural['train']['y'], neural_weight)))
    vx, vy = (np.concatenate([row[key] for row in primary]) for key in ('x', 'y'))
    groups = np.asarray([domain * 8 + PROFILES.index(profile) for domain, row in enumerate(primary) for profile in row['profiles']], dtype=np.int64)
    result = fit(x, y, vx, vy, groups, epochs=epochs, hidden=hidden, seed=42, feature_count=261)
    layers = result.pop('layers')
    calibrated = calibrate(vy, scores(vx, layers), groups,
                           initial_threshold=result['checkpoint_threshold_q10'], max_passes=max_passes)
    table = np.asarray(calibrated['thresholds_q10'], dtype=np.int32)
    familiar = neural['validation']
    fp = np.asarray([PROFILES.index(p) for p in familiar['profiles']])
    familiar_predictions = decisions(scores(familiar['x'], layers), table)
    familiar_rows = [dict(domain='familiar-neural', profile=profile,
                          **measure(familiar['y'][fp == i], familiar_predictions[fp == i])) for i, profile in enumerate(PROFILES)]
    rows = [dict(row, domain='legacy' if row['domain'] == 'legacy' else 'held-neural')
            for row in calibrated['domain_profiles']] + familiar_rows
    all_y = np.concatenate((vy, familiar['y']))
    all_predictions = np.concatenate((decisions(scores(vx, layers), table), familiar_predictions))
    all_profiles = np.concatenate((groups % 8, fp))
    pooled = [dict(profile=profile, **measure(all_y[all_profiles == i], all_predictions[all_profiles == i]))
              for i, profile in enumerate(PROFILES)]
    if before != _input_hashes(paths) or code != _source_hashes():
        raise ValueError('development data or source changed during fit')
    result.update(scope='development_caption_holdout_training_not_final', frontend='context-duration-cmn',
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
                  weights_bytes=sum(r['weight'].nbytes + r['bias'].nbytes + 4 for r in layers),
                  frozen=False, final_evaluated=False, device_tested=False, real_voice_tested=False, vast_usd=0)
    output.mkdir(parents=True, mode=0o700)
    np.savez(output / 'weights.npz', **{f'{field}{i}': layer[field] for i, layer in enumerate(layers) for field in ('weight', 'bias', 'shift')})
    result['weights_sha256'] = digest(output / 'weights.npz')
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
