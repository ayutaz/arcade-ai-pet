"""Strict numeric assets for 261-input context/duration models; not a freeze."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile
import numpy as np
from .native import _array, _render_tables

CONTRACT = dict(conditioning='hp160', frontend='context-duration', peak=4096,
                samples=32000, rate=16000, features=261, context_frames=10, duration_scale=4)
RUNTIME = 'context-duration-class-v1'


def real_path(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('real path without symlink ancestors required')
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thresholds(values):
    if not isinstance(values, (list, tuple)) or len(values) != 5 or any(type(x) is not int or not 0 <= x <= 4096 for x in values):
        raise ValueError('five integer thresholds in 0..4096 required')
    return list(values)


def read_layers(path):
    path = real_path(path)
    if not path.is_file() or path.stat().st_size > 262144:
        raise ValueError('bounded regular model archive required')
    expected = {f'{name}{i}.npy' for i in (0, 1) for name in ('weight', 'bias', 'shift')}
    headers = {}
    try:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) != 6 or set(archive.namelist()) != expected:
                raise ValueError('exactly six integer model members required')
            for info in archive.infolist():
                if info.file_size > 65536:
                    raise ValueError('model array budget')
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=2048)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=2048)
                    else:
                        raise ValueError('model array format')
                    if fortran or stream.tell() + math.prod(shape) * dtype.itemsize != info.file_size:
                        raise ValueError('model shape/payload mismatch')
                    if info.filename.startswith('weight'):
                        valid = dtype == np.int8 and len(shape) == 2 and 1 <= shape[0] <= 90 and 1 <= shape[1] <= 261
                    elif info.filename.startswith('bias'):
                        valid = dtype == np.int32 and len(shape) == 1 and 1 <= shape[0] <= 90
                    else:
                        valid = np.issubdtype(dtype, np.integer) and dtype.itemsize <= 8 and shape == ()
                        if valid:
                            value = int.from_bytes(stream.read(dtype.itemsize),
                                                   'big' if dtype.byteorder == '>' else 'little', signed=dtype.kind == 'i')
                            valid = 0 <= value <= 16
                    if not valid:
                        raise ValueError('bounded integer representation required')
                    headers[info.filename] = shape
        hidden = headers['weight0.npy'][0]
        required = {'weight0.npy': (hidden, 261), 'bias0.npy': (hidden,), 'shift0.npy': (),
                    'weight1.npy': (7, hidden), 'bias1.npy': (7,), 'shift1.npy': ()}
        if headers != required:
            raise ValueError('261-input two-layer architecture required')
        with np.load(path, allow_pickle=False) as archive:
            layers = [dict(weight=archive[f'weight{i}'], bias=archive[f'bias{i}'], shift=int(archive[f'shift{i}'])) for i in (0, 1)]
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise ValueError('readable bounded model archive required') from exc
    size = sum(row['weight'].nbytes + row['bias'].nbytes + 4 for row in layers)
    if size + 20 > 24576:
        raise ValueError('24KiB numeric budget including thresholds')
    return layers, size


def render(layers, table):
    table = thresholds(table)
    tables, _ = _render_tables(layers, 0)
    # Model provenance/licensing belongs to the dataset/model card, separately
    # from this Apache-2.0 exporter. Do not inherit the legacy HTS-only notice.
    model = '/* Generated numeric model. Data/model licensing is separate from code. */\n'
    model += f'enum {{ PET_DECISION_HIDDEN={layers[0]["weight"].shape[0]} }};\n'
    for i, row in enumerate(layers):
        model += f'enum {{ PET_DECISION_SHIFT{i}={row["shift"]} }};\n'
        model += _array(f'pet_decision_weight{i}', row['weight'], 'int8_t')
        model += _array(f'pet_decision_bias{i}', row['bias'], 'int32_t')
    model += _array('pet_decision_thresholds', np.asarray(table, dtype=np.int32), 'int32_t')
    return {'feature_tables.h': tables, 'decision_model.h': model}


def export_assets(weights, table, output):
    table = thresholds(table)
    weights = real_path(weights)
    output = real_path(output)
    if output.exists():
        raise ValueError('fresh numeric asset output required')
    layers, size = read_layers(weights)
    rendered = render(layers, table)
    result = dict(runtime=RUNTIME, input_contract=CONTRACT.copy(), thresholds_q10=table,
                  weights_sha256=digest(weights), weights_bytes=size, numeric_bytes=size + 20,
                  header_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in rendered.items()},
                  frozen=False, device_tested=False, real_voice_tested=False)
    output.mkdir(parents=True, mode=0o700)
    for name, text in rendered.items():
        (output / name).write_text(text)
    shutil.copyfile(weights, output / 'weights.npz')
    (output / 'decision-metadata.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result
