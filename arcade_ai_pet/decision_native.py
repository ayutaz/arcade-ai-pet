"""Trusted C conformance runtime for new numeric assets, without deployment."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from . import decision_assets as assets
from .channel_native import ChannelNativeModel

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('decision_features.c', 'channel_prepare.c', 'decision_inference.c',
           'decision_features.h', 'channel_features.h', 'pet_kws.h')


class DecisionNativeModel(ChannelNativeModel):
    def __init__(self, directory):
        directory = assets.real_path(directory)
        meta = json.loads(assets.real_path(directory / 'decision-metadata.json').read_text())
        table = assets.thresholds(meta['thresholds_q10'])
        self.layers, size = assets.read_layers(directory / 'weights.npz')
        if (meta['runtime'] != assets.RUNTIME or meta['input_contract'] != assets.CONTRACT or
            assets.digest(directory / 'weights.npz') != meta['weights_sha256'] or
            size != meta['weights_bytes'] or size + 20 != meta['numeric_bytes'] or meta['frozen'] is not False):
            raise ValueError('new numeric model identity/contract mismatch')
        rendered = assets.render(self.layers, table)
        for name, text in rendered.items():
            expected = hashlib.sha256(text.encode()).hexdigest()
            if assets.digest(assets.real_path(directory / name)) != expected or meta['header_sha256'].get(name) != expected:
                raise ValueError('header differs from validated numeric export')
        self.temp = tempfile.TemporaryDirectory()
        self.weights_bytes = size
        self.metadata = meta
        try:
            base = Path(self.temp.name)
            for name, text in rendered.items():
                (base / name).write_text(text)
            subprocess.run(['clang', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                            '-I' + str(ROOT / 'native'), '-I' + str(base),
                            *[str(ROOT / 'native' / name) for name in SOURCES[:3]], '-lm', '-o', str(base / 'decision.so')],
                           capture_output=True, check=True, timeout=30)
            self.lib = C.CDLL(str(base / 'decision.so'))
            self.lib.pet_decision_feature_scratch_size.restype = C.c_size_t
            self.scratch_bytes = self.lib.pet_decision_feature_scratch_size()
            if self.scratch_bytes > 24576:
                raise ValueError('feature scratch budget')
            self.scratch = C.create_string_buffer(self.scratch_bytes)
            for name in ('pet_channel_prepare', 'pet_channel_prepare_banked'):
                getattr(self.lib, name).argtypes = [C.c_void_p, C.c_size_t]
            for name in ('pet_decision_features', 'pet_decision_features_banked'):
                getattr(self.lib, name).argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
            for name in ('pet_decision_predict', 'pet_decision_classify'):
                getattr(self.lib, name).argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
            self.source_sha256 = {name: assets.digest(ROOT / 'native' / name) for name in SOURCES}
        except BaseException:
            self.temp.cleanup()
            raise

    def extract(self, pcm, *, banked=False):
        values, banks, pointer = self._prepared(pcm, banked)
        result = np.zeros(261, dtype=np.float32)
        fn = self.lib.pet_decision_features_banked if banked else self.lib.pet_decision_features
        if fn(pointer if banked else values.ctypes.data, 32000, result.ctypes.data, 261, self.scratch, self.scratch_bytes) != 1:
            raise ValueError('context-duration features failed')
        return result

    def predict(self, values):
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != (261,) or not np.isfinite(values).all():
            raise ValueError('finite 261 float32 features required')
        values = np.ascontiguousarray(values)
        logits = np.zeros(7, dtype=np.int32)
        label = self.lib.pet_decision_predict(values.ctypes.data, 261, logits.ctypes.data, 7)
        if not 0 <= label <= 6:
            raise ValueError('new integer inference failed')
        return label, logits
