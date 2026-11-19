"""Reserved four-condition generation and evaluation for the frozen new runtime."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from . import decision_assets as assets
from .decision_freeze import verify, _features, _recompute, _accepted
from .channel_robustness import PROFILES, measure
from .native_evaluation import _examples
from .neural_corpus import validate_plan, _generate, _finalize, _digest
from .data import SOURCES
from .features import split_for
from .training import scores
from .class_rejection import decisions


def binding(frozen):
    frozen = assets.real_path(frozen)
    meta, _ = verify(frozen)
    return dict(weights_sha256=meta['weights_sha256'], plan_sha256=meta['plan_sha256'],
                frozen_sha256=assets.digest(frozen / 'decision-frozen.json'))


def generate_final(plan, output, *, frozen, synthesize, recognize):
    bound = binding(frozen)
    if validate_plan(plan) != bound['plan_sha256']:
        raise ValueError('only the frozen reserved plan can generate final sources')
    return _generate(plan, output, synthesize=synthesize, recognize=recognize, finalize=False,
                     role='final', model_binding=bound)


def finalize_final(output, *, frozen):
    output = assets.real_path(output)
    bound = binding(frozen)
    qualified = json.loads(assets.real_path(output / 'qualified-sources.json').read_text())
    plan = json.loads(assets.real_path(output / 'plan.json').read_text())
    meta, _ = verify(frozen)
    if (qualified.get('model_binding') != bound or validate_plan(plan) != bound['plan_sha256'] or
        {r['condition_id'] for r in qualified['records']} != set(meta['final_condition_ids'])):
        raise ValueError('final generation binding or condition coverage changed')
    return _finalize(output, role='final')


def _native_audit(frozen, dataset, layers, table):
    meta, _ = verify(frozen)
    _, _, result = _recompute({'final': dataset}, frozen, layers, table, validation_only=False, runtime=meta['runtime'])
    return result


def _validate_manifest(manifest, meta, plan):
    if (type(manifest.get('variants')) is not int or manifest['variants'] != plan['variants'] or
        manifest.get('provider') != plan['provider'] or
        manifest.get('generator_sha256') != meta['source_sha256']['arcade_ai_pet/neural_corpus.py'] or
        manifest.get('audit_sha256') != meta['source_sha256']['arcade_ai_pet/generated_audio.py']):
        raise ValueError('final generation provenance changed')
    seen, waves = set(), set()
    for row in manifest['records']:
        source = row.get('source'); ident = row.get('id'); label = row.get('label')
        if (type(source) is not str or type(label) is not int or not 0 <= label <= 6 or
            hashlib.sha256(source.encode()).hexdigest() != ident or ident in seen or row.get('split') != split_for(source)):
            raise ValueError('final source identity changed')
        seen.add(ident)
        if 'sha256' in row:
            condition = row.get('condition_id'); prompt = row.get('prompt'); wave = row['sha256']
            if (condition not in meta['final_condition_ids'] or prompt not in SOURCES.get(label, ()) or
                not isinstance(wave, str) or not re.fullmatch('[0-9a-f]{64}', wave) or wave in waves):
                raise ValueError('final speech provenance changed')
            prefix = f"neural/{meta['plan_sha256']}/{condition}/{label}/{_digest(prompt)}/"
            if not source.startswith(prefix) or source[len(prefix):] not in map(str, range(plan['attempts'])):
                raise ValueError('final speech relabelled under a different condition')
            waves.add(wave)
        elif label != 6 or not re.fullmatch('neural-final-' + meta['plan_sha256'][:12] + r'/[0-9]{1,3}', source) or int(source.rsplit('/', 1)[1]) >= 200:
            raise ValueError('final control source changed')


def evaluate_final(frozen, dataset, output):
    frozen, dataset, output = map(assets.real_path, (frozen, dataset, output))
    if output.exists():
        raise ValueError('fresh final report output required')
    meta, layers = verify(frozen); bound = binding(frozen)
    manifest_hash = assets.digest(assets.real_path(dataset / 'manifest.json'))
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if (manifest.get('role') != 'final' or manifest.get('model_binding') != bound or
        manifest.get('plan_sha256') != meta['plan_sha256'] or not manifest.get('records')):
        raise ValueError('final corpus/model binding required')
    ids = {r['id'] for r in manifest['records']}; waves = {r['sha256'] for r in manifest['records'] if 'sha256' in r}
    conditions = {r.get('condition_id') for r in manifest['records'] if 'sha256' in r}
    if (ids & set(meta['source_ids']) or waves & set(meta['wave_hashes']) or
        conditions != set(meta['final_condition_ids'])):
        raise ValueError('source/wave overlap or missing reserved final condition')
    plan = json.loads((frozen / 'final-plan.json').read_text())
    _validate_manifest(manifest, meta, plan)
    table = np.asarray(meta['thresholds_q10'], dtype=np.int32)
    expanded = [r for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
    labels, predictions = [], {p: [] for p in PROFILES}
    for (pcm, label, _), row in zip(_examples(dataset), expanded, strict=True):
        if row['label'] != label:
            raise ValueError('final reconstructed label changed')
        labels.append(label); seed = int(row['id'][:16], 16)
        for profile in PROFILES:
            logit = scores(_features(pcm, profile, seed, meta['runtime'])[None, :], layers)
            predictions[profile].append(int(decisions(logit, table)[0]))
    if set(labels) != set(range(7)):
        raise ValueError('all seven final classes required')
    rows = [dict(profile=p, **measure(labels, predictions[p])) for p in PROFILES]
    native = _native_audit(frozen, dataset, layers, table)
    # Detect any mutation during this one evaluation, including source or plan.
    if binding(frozen) != bound or assets.digest(dataset / 'manifest.json') != manifest_hash:
        raise ValueError('freeze or manifest changed during final evaluation')
    result = dict(schema=1, scope='reserved_synthetic_final_host_only', model_binding=bound, runtime=meta['runtime'],
                  weights_sha256=meta['weights_sha256'], thresholds_q10=meta['thresholds_q10'],
                  dataset_manifest_sha256=manifest_hash, profiles=rows,
                  native_conformance=native, accepted_synthetic=_accepted(rows) and native['matched'],
                  source_overlap=0, wave_overlap=0, final_evaluated=True, model_selection_used_final=False,
                  device_tested=False, real_voice_tested=False, vast_usd=0)
    output.mkdir(parents=True)
    (output / 'decision-final-evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
