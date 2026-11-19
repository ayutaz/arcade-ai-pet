"""Generate reserved synthetic conditions only after a verified model freeze."""
import hashlib,json
from pathlib import Path
from .channel_freeze import _verify
from .neural_corpus import validate_plan,_generate,_finalize


def _binding(frozen):
    frozen=Path(frozen).absolute()
    if frozen.resolve()!=frozen:raise ValueError('real frozen model directory required')
    metadata,_=_verify(frozen)
    return dict(weights_sha256=metadata['weights_sha256'],
                frozen_sha256=hashlib.sha256((frozen/'channel-frozen.json').read_bytes()).hexdigest())


def generate_final(plan,output,*,frozen,synthesize,recognize):
    validate_plan(plan)
    binding=_binding(frozen)
    return _generate(plan,output,synthesize=synthesize,recognize=recognize,
                     finalize=False,role='final',model_binding=binding)


def finalize_final(output,*,frozen):
    output=Path(output).absolute();binding=_binding(frozen)
    qualified=json.loads((output/'qualified-sources.json').read_text())
    if qualified.get('model_binding')!=binding:
        raise ValueError('model freeze changed after final generation')
    return _finalize(output,role='final')
