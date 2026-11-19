"""Freeze the complete 261-feature decision contract before final generation."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from . import decision_assets as assets
from .decision_native import DecisionNativeModel, SOURCES as NATIVE_SOURCES
from .channel_freeze import SOURCES as LEGACY_SOURCES
from .channel_robustness import PROFILES, simulate, condition, measure
from .channel_training import level
from .duration_frontend import features
from .native_evaluation import _examples
from .neural_corpus import validate_plan
from .training import scores
from .class_rejection import decisions

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(dict.fromkeys((*LEGACY_SOURCES,
    'arcade_ai_pet/decision_assets.py', 'arcade_ai_pet/decision_native.py',
    'arcade_ai_pet/decision_freeze.py', 'arcade_ai_pet/decision_final.py',
    'arcade_ai_pet/duration_frontend.py', 'arcade_ai_pet/context_frontend.py',
    'arcade_ai_pet/class_rejection.py', 'arcade_ai_pet/domain_training.py',
    'arcade_ai_pet/native.py', 'arcade_ai_pet/neural_corpus.py', 'arcade_ai_pet/generated_audio.py',
    *('native/' + name for name in NATIVE_SOURCES))))


def _source_hashes():
    return {name: assets.digest(ROOT / name) for name in SOURCES}


def _accepted(rows):
    return bool(rows) and all(r['macro_f1'] >= .90 and r['false_activation_rate'] <= .05 and r['command_recall'] >= .80 for r in rows)


def _features(pcm, profile, seed):
    return features(level(condition(simulate(pcm, profile, seed), 'hp160')), context=True)


def _domains(domains, final_ids):
    if (type(domains) is not dict or not 2 <= len(domains) <= 4 or
        any(type(name) is not str or not re.fullmatch('[a-z][a-z0-9-]{0,31}', name) for name in domains)):
        raise ValueError('two to four named development domains required')
    paths = {name: assets.real_path(path) for name, path in domains.items()}
    ids, waves, manifests = set(), set(), {}
    for name, path in paths.items():
        manifest = json.loads(assets.real_path(path / 'manifest.json').read_text())
        if manifest.get('role', 'development') != 'development' or not manifest.get('records'):
            raise ValueError('nonempty development-only domains required')
        for row in manifest['records']:
            ident = row['id']; wave = row.get('sha256')
            if (not isinstance(ident, str) or not re.fullmatch('[0-9a-f]{64}', ident) or ident in ids or
                wave is not None and (not isinstance(wave, str) or not re.fullmatch('[0-9a-f]{64}', wave) or wave in waves) or
                row.get('condition_id') in final_ids):
                raise ValueError('overlapping development sources or reserved final conditions')
            ids.add(ident)
            if wave is not None:
                waves.add(wave)
        manifests[name] = assets.digest(path / 'manifest.json')
    return paths, sorted(ids), sorted(waves), manifests


def _recompute(paths, directory, layers, table, *, validation_only):
    labels, predictions, groups = [], [], []
    stats = dict(rows=0, exact_integer_logits=0, banked_features_equal=0, final_decisions_equal=0,
                 maximum_feature_error=0.0)
    with DecisionNativeModel(directory) as engine:
        for domain_index, (name, dataset) in enumerate(paths.items()):
            manifest = json.loads((dataset / 'manifest.json').read_text())
            expanded = [r for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
            for (pcm, label, _), row in zip(_examples(dataset), expanded, strict=True):
                if label != row['label']:
                    raise ValueError('reconstructed label changed')
                if validation_only and row['split'] != 'validation':
                    continue
                seed = int(row['id'][:16], 16)
                for profile_index, profile in enumerate(PROFILES):
                    expected = _features(pcm, profile, seed)
                    logits = scores(expected[None, :], layers)[0]
                    prediction = int(decisions(logits[None, :], table)[0])
                    changed = simulate(pcm, profile, seed)
                    native = engine.extract(changed)
                    banked = engine.extract(changed, banked=True)
                    direct, native_logits = engine.predict(expected)
                    end, _ = engine.predict(native)
                    stats['rows'] += 1
                    stats['exact_integer_logits'] += int(np.array_equal(logits, native_logits) and direct == prediction)
                    stats['banked_features_equal'] += int(np.array_equal(native, banked))
                    stats['final_decisions_equal'] += int(end == prediction)
                    stats['maximum_feature_error'] = max(stats['maximum_feature_error'], float(abs(native - expected).max()))
                    labels.append(label); predictions.append(prediction); groups.append(domain_index * 8 + profile_index)
        stats['scratch_bytes'] = engine.scratch_bytes
    y, p, g = np.asarray(labels), np.asarray(predictions), np.asarray(groups)
    if not len(y) or set(g.tolist()) != set(range(len(paths) * 8)):
        raise ValueError('every domain/profile must have validation examples')
    rows = [dict(domain=name, profile=profile, **measure(y[g == i * 8 + j], p[g == i * 8 + j]))
            for i, name in enumerate(paths) for j, profile in enumerate(PROFILES)]
    pooled = [dict(profile=profile, **measure(y[g % 8 == j], p[g % 8 == j])) for j, profile in enumerate(PROFILES)]
    stats['matched'] = (all(stats[key] == stats['rows'] for key in ('exact_integer_logits', 'banked_features_equal', 'final_decisions_equal'))
                        and stats['maximum_feature_error'] < .001 and stats['scratch_bytes'] <= 24576)
    return rows, pooled, stats


def freeze_candidate(weights, table, domains, plan, output):
    output = assets.real_path(output)
    if output.exists():
        raise ValueError('fresh frozen output required')
    plan_hash = validate_plan(plan)
    final_ids = [r['id'] for r in plan['conditions'] if r['role'] == 'final']
    if len(final_ids) != 4:
        raise ValueError('four reserved final conditions required')
    table = assets.thresholds(table)
    weights = assets.real_path(weights)
    layers, size = assets.read_layers(weights)
    paths, ids, waves, manifests = _domains(domains, set(final_ids))
    code_hashes = _source_hashes(); weights_hash = assets.digest(weights)
    # A failed gate leaves the requested output absent. Temporary generated
    # headers are validated/regenerated again by the trusted native constructor.
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp).resolve() / 'assets'
        asset_meta = assets.export_assets(weights, table, temporary)
        rows, pooled, conformance = _recompute(paths, temporary, layers, np.asarray(table, dtype=np.int32), validation_only=True)
        if not _accepted(rows) or not _accepted(pooled) or not conformance['matched']:
            raise ValueError('development quality/native gates failed')
        if (code_hashes != _source_hashes() or weights_hash != assets.digest(weights) or
            manifests != {name: assets.digest(path / 'manifest.json') for name, path in paths.items()}):
            raise ValueError('inputs changed during freeze')
        result = dict(schema=1, runtime=assets.RUNTIME, input_contract=assets.CONTRACT.copy(),
                      architecture=[261, layers[0]['weight'].shape[0], 7], thresholds_q10=table,
                      weights_sha256=weights_hash, weights_bytes=size, numeric_bytes=size + 20,
                      source_sha256=code_hashes, development_manifest_sha256=manifests,
                      source_ids=ids, wave_hashes=waves, plan_sha256=plan_hash, final_condition_ids=final_ids,
                      domain_profiles=rows, pooled_profiles=pooled, conformance=conformance,
                      asset_metadata_sha256=assets.digest(temporary / 'decision-metadata.json'),
                      header_sha256=asset_meta['header_sha256'], final_evaluated=False,
                      device_tested=False, real_voice_tested=False,
                      license='model/data conditions are separate; see dataset provenance and docs/data-license.md')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temporary, output)
    (output / 'final-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (output / 'decision-frozen.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(directory):
    directory = assets.real_path(directory)
    meta = json.loads(assets.real_path(directory / 'decision-frozen.json').read_text())
    table = assets.thresholds(meta['thresholds_q10'])
    if (meta.get('schema') != 1 or meta.get('runtime') != assets.RUNTIME or
        meta.get('input_contract') != assets.CONTRACT or meta.get('source_sha256') != _source_hashes()):
        raise ValueError('frozen source/contract changed')
    layers, size = assets.read_layers(directory / 'weights.npz')
    if (assets.digest(directory / 'weights.npz') != meta['weights_sha256'] or size != meta['weights_bytes'] or
        size + 20 != meta['numeric_bytes'] or meta['architecture'] != [261, layers[0]['weight'].shape[0], 7]):
        raise ValueError('frozen numeric model changed')
    plan = json.loads(assets.real_path(directory / 'final-plan.json').read_text())
    if (validate_plan(plan) != meta['plan_sha256'] or
        [r['id'] for r in plan['conditions'] if r['role'] == 'final'] != meta['final_condition_ids'] or
        len(meta['final_condition_ids']) != 4):
        raise ValueError('reserved final plan changed')
    if assets.digest(assets.real_path(directory / 'decision-metadata.json')) != meta['asset_metadata_sha256']:
        raise ValueError('frozen numeric metadata changed')
    numeric = json.loads((directory / 'decision-metadata.json').read_text())
    if numeric['thresholds_q10'] != table or numeric['weights_sha256'] != meta['weights_sha256']:
        raise ValueError('frozen table/numeric identity changed')
    for name in ('feature_tables.h', 'decision_model.h'):
        if assets.digest(assets.real_path(directory / name)) != meta['header_sha256'][name]:
            raise ValueError('frozen headers changed')
    return meta, layers
