"""Same development rows/augmentation as context-duration, separately centered."""
import json
from pathlib import Path
from .duration_arrays import make_arrays as original
from .cmn_frontend import center


def make_arrays(dataset, candidate, *, frontend='context-duration-cmn', noise_replicas=1,
                noise_mode='fresh', hum_replicas=1):
    if candidate != 'hp160' or frontend != 'context-duration-cmn':
        raise ValueError('explicit hp160/context-duration-cmn required')
    dataset = Path(dataset)
    if json.loads((dataset / 'manifest.json').read_text()).get('role', 'development') != 'development':
        raise ValueError('final CMN corpus forbidden')
    arrays = original(dataset, candidate, frontend='context-duration', noise_replicas=noise_replicas,
                      noise_mode=noise_mode, hum_replicas=hum_replicas)
    if set(arrays) != {'train', 'validation'}:
        raise ValueError('only development train/validation arrays allowed')
    return {split: dict(row, x=center(row['x'])) for split, row in arrays.items()}
