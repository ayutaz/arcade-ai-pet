"""ASR-qualified train-only coupled speed/pitch perturbation; no truncation."""
import hashlib
import json
from pathlib import Path
import re
import wave
import numpy as np
from .channel_training import make_arrays as adaptive_arrays, level
from .channel_robustness import PROFILES, simulate, condition
from .channel_frontend import features
from .data import read_pcm, trim_silence, augment
from .generated_audio import PROMPT_LABELS, audit_generated_wave, _read_wave

RATES = (.9, 1.1)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def _manifest(dataset):
    dataset = Path(dataset).absolute()
    if dataset.resolve() != dataset:
        raise ValueError('real development dataset required')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if (manifest.get('role') != 'development' or type(manifest.get('variants')) is not int or
            not 1 <= manifest['variants'] <= 10):
        raise ValueError('bounded development dataset required')
    return dataset, manifest


def _sources(dataset, manifest):
    sources = {}
    for row in manifest['records']:
        if row['split'] != 'train' or 'sha256' not in row or row['label'] >= 6:
            continue
        ident = row['id']
        if (type(ident) is not str or not re.fullmatch('[0-9a-f]{64}', ident) or
                ident in sources or PROMPT_LABELS.get(row.get('prompt')) != row['label']):
            raise ValueError('unique train prompt identity required')
        path = dataset/'sources'/(ident+'.wav')
        raw, rate, pcm = _read_wave(path)
        if rate != 16000 or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('training source changed')
        sources[ident] = (row, trim_silence(pcm))
    if not sources:
        raise ValueError('spoken training sources required')
    return sources


def speed_variant(pcm, rate):
    if (not isinstance(pcm, np.ndarray) or pcm.dtype != np.int16 or pcm.ndim != 1 or
            not 1 <= len(pcm) <= 32000 or type(rate) is not float or rate not in RATES):
        raise ValueError('bounded PCM16 and explicit training rate required')
    count = round(len(pcm)/rate)
    if not 1 <= count <= 32000:
        raise ValueError('rate variant exceeds two seconds; do not truncate')
    values = np.interp(np.linspace(0, len(pcm)-1, count), np.arange(len(pcm)), pcm)
    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


def prepare_sources(dataset, output):
    dataset, manifest = _manifest(dataset)
    output = Path(output).absolute()
    if output.resolve() != output or output.exists():
        raise ValueError('fresh real output required')
    sources = _sources(dataset, manifest)
    output.mkdir(parents=True)
    (output/'sources').mkdir()
    rows = []
    skipped = 0
    for ident, (row, pcm) in sources.items():
        for rate in RATES:
            if round(len(pcm)/rate) > 32000:
                skipped += 1
                continue
            changed = speed_variant(pcm, rate)
            name = hashlib.sha256(f'prosody-v1/{ident}/{rate}'.encode()).hexdigest()+'.wav'
            path = output/'sources'/name
            with wave.open(str(path), 'wb') as stream:
                stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                stream.writeframes(changed.astype('<i2').tobytes())
            rows.append(dict(source_id=ident, source_sha256=row['sha256'], prompt=row['prompt'],
                             label=row['label'], rate=rate, file=name, wave_sha256=_sha(path), frames=len(changed)))
    plan = dict(schema=1, role='development_train_only', dataset_manifest_sha256=_sha(dataset/'manifest.json'),
                generator_sha256=_sha(__file__), rates=list(RATES), skipped_long_transforms=skipped, transforms=rows)
    _write(output/'plan.json', plan)
    return plan


def _plan(folder):
    folder = Path(folder).absolute()
    if folder.resolve() != folder:
        raise ValueError('real qualification folder required')
    plan = json.loads((folder/'plan.json').read_text())
    if (plan.get('schema') != 1 or plan.get('role') != 'development_train_only' or
            plan.get('generator_sha256') != _sha(__file__) or plan.get('rates') != list(RATES)):
        raise ValueError('qualification generator identity changed')
    seen = set()
    for row in plan['transforms']:
        name = row['file']
        if (not re.fullmatch('[0-9a-f]{64}\\.wav', name) or name in seen or
                row['rate'] not in RATES or PROMPT_LABELS.get(row['prompt']) != row['label']):
            raise ValueError('unique qualified transform identity required')
        raw, rate, pcm = _read_wave(folder/'sources'/name)
        if rate != 16000 or len(pcm) != row['frames'] or hashlib.sha256(raw).hexdigest() != row['wave_sha256']:
            raise ValueError('qualified transform changed')
        seen.add(name)
    return folder, plan


def qualify_sources(folder, *, recognize):
    folder, plan = _plan(folder)
    if (folder/'qualification.json').exists() or not callable(recognize):
        raise ValueError('fresh qualification and recognizer required')
    rows = []
    for row in plan['transforms']:
        audit = audit_generated_wave(folder/'sources'/row['file'], prompt=row['prompt'], recognize=recognize)
        rows.append(dict(file=row['file'], **audit))
        (folder/'qualification-progress.json').write_text(json.dumps(dict(rows=rows), ensure_ascii=False, indent=2)+'\n')
    accepted = sum(row['content_check_passed'] for row in rows)
    result = dict(schema=1, role='development_train_only', plan_sha256=_sha(folder/'plan.json'),
                  audit_sha256=_sha(Path(__file__).with_name('generated_audio.py')),
                  complete=all(row['status'] != 'recognizer_error' for row in rows), rows=rows,
                  accepted_transforms=accepted, rejected_transforms=len(rows)-accepted,
                  independent_sources_added=0, human_transcript_verified=False)
    _write(folder/'qualification.json', result)
    return result


def make_arrays(dataset, candidate, *, frontend='adaptive', noise_replicas=1,
                noise_mode='fresh', hum_replicas=1, qualified=None):
    dataset, manifest = _manifest(dataset)
    if candidate != 'hp160' or frontend != 'adaptive' or qualified is None:
        raise ValueError('explicit qualified adaptive development augmentation required')
    folder, plan = _plan(qualified)
    report = json.loads((folder/'qualification.json').read_text())
    if (plan['dataset_manifest_sha256'] != _sha(dataset/'manifest.json') or
            report.get('schema') != 1 or report.get('role') != 'development_train_only' or
            report.get('complete') is not True or report.get('plan_sha256') != _sha(folder/'plan.json') or
            report.get('audit_sha256') != _sha(Path(__file__).with_name('generated_audio.py')) or
            len(report['rows']) != len(plan['transforms'])):
        raise ValueError('complete unchanged source qualification required')
    sources = _sources(dataset, manifest)
    accepted = []
    for row, audit in zip(plan['transforms'], report['rows']):
        source = sources.get(row['source_id'])
        if (source is None or any(source[0][key] != row[value] for key, value in
                (('sha256','source_sha256'), ('prompt','prompt'), ('label','label'))) or
                any(audit.get(key) != row[key] for key in ('file','prompt','label','wave_sha256','frames')) or
                audit.get('status') not in ('matched','mismatch') or
                audit.get('content_check_passed') is not (audit['status'] == 'matched')):
            raise ValueError('qualified training identity differs')
        if audit['content_check_passed']:
            accepted.append(row)
    if (len(accepted) != report['accepted_transforms'] or
            len(plan['transforms'])-len(accepted) != report['rejected_transforms']):
        raise ValueError('qualification counts differ')
    base = adaptive_arrays(dataset, candidate, frontend=frontend, noise_replicas=noise_replicas,
                           noise_mode=noise_mode, hum_replicas=hum_replicas)
    xs, ys, profiles = [], [], []
    for row in accepted:
        # Read the exact PCM that ASR checked; do not regenerate with another NumPy version.
        changed = read_pcm(folder/'sources'/row['file'])
        for variant in range(manifest['variants']):
            key = f"prosody-v1/{row['source_id']}/{row['rate']}/{variant}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
            pcm = augment(changed, seed)
            for profile in PROFILES:
                replicas = noise_replicas if profile == 'noise' else hum_replicas if profile == 'hum' else 1
                for replica in range(replicas):
                    current = int(row['source_id'][:16], 16)
                    if replica and profile == 'noise' and noise_mode == 'fresh':
                        current = int.from_bytes(hashlib.sha256(row['source_id'].encode()+pcm.tobytes()+bytes([replica])).digest()[:8], 'little')
                    xs.append(features(level(condition(simulate(pcm, profile, current), candidate))))
                    ys.append(row['label'])
                    profiles.append(profile)
    train = dict(x=np.concatenate([base['train']['x'], np.asarray(xs, dtype=np.float32).reshape(-1, 260)]),
                 y=np.concatenate([base['train']['y'], np.asarray(ys, dtype=np.int64)]),
                 profiles=list(base['train']['profiles'])+profiles)
    return dict(train=train, validation=base['validation'], augmentation=dict(
        kind='coupled-speed-pitch-v1', rates=list(RATES), independent_sources_added=0,
        added_training_rows=len(ys), source_rate_transforms=len(plan['transforms']),
        accepted_transforms=len(accepted), rejected_transforms=report['rejected_transforms'],
        qualification_sha256=_sha(folder/'qualification.json'),
        skipped_long_transforms=plan['skipped_long_transforms'], validation_changed=False))
