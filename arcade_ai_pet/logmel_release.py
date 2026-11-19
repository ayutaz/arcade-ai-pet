"""Load only the published log-mel v1 freeze, using its byte-exact math tables.

Formula regeneration can differ between NumPy/CPU math implementations. The
frozen source, headers, model, metadata and plan remain mandatory hash bindings;
this loader does not accept approximately equal or newly generated tables.
"""
import ctypes as C
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from . import dscnn_assets as assets,dscnn_freeze as frozen
from .dscnn_native import DscnnNativeModel
from .logmel_native import LogmelNativeFrontend,SOURCES
from .neural_corpus import validate_plan

ROOT=Path(__file__).resolve().parents[1]
FROZEN_SHA256='89e06e7eda40c3af46c12883e3cbb77415960b466683094ad787d406b1330ab6'
TABLES_SHA256='bd9ff5df6f6033179122c5f957bb6cbeaf1445d643dc1ef861153f6bfa0da110'


def verify_release(directory):
    directory=assets.real_path(directory)
    path=assets.real_path(directory/'decision-frozen.json')
    if assets.digest(path)!=FROZEN_SHA256:raise ValueError('published frozen identity changed')
    meta=json.loads(path.read_text())
    if meta['source_sha256']!=frozen.source_hashes():raise ValueError('frozen source changed')
    if (assets.digest(assets.real_path(directory/'weights.npz'))!=meta['weights_sha256'] or
        assets.digest(assets.real_path(directory/'decision-metadata.json'))!=meta['asset_metadata_sha256']):
        raise ValueError('frozen numeric identity changed')
    plan=json.loads(assets.real_path(directory/'final-plan.json').read_text())
    if validate_plan(plan)!=meta['plan_sha256']:raise ValueError('frozen final plan changed')
    layers,size=assets.read_layers(directory/'weights.npz');table=assets.thresholds(meta['thresholds_q10'])
    if meta['weights_bytes']!=size or meta['numeric_bytes']!=size+20:raise ValueError('frozen numeric size changed')
    if meta['header_sha256']['logmel_tables.h']!=TABLES_SHA256:raise ValueError('canonical tables required')
    for name,wanted in meta['header_sha256'].items():
        if assets.digest(assets.real_path(directory/name))!=wanted:raise ValueError('frozen header changed')
    # Learned integer arrays regenerate identically. Math tables are verified by
    # their fixed SHA above, not replaced with the local formula's different bits.
    text=assets.render(layers,table)['logmel_model.h']
    if hashlib.sha256(text.encode()).hexdigest()!=meta['header_sha256']['logmel_model.h']:
        raise ValueError('frozen learned integer header changed')
    return meta,layers


class _FrozenFrontend(LogmelNativeFrontend):
    def __init__(self,tables):
        if not isinstance(tables,bytes) or hashlib.sha256(tables).hexdigest()!=TABLES_SHA256:
            raise ValueError('byte-exact canonical math tables required')
        self.tables_sha256=TABLES_SHA256
        self.temp=tempfile.TemporaryDirectory();self.closed=False
        try:
            base=Path(self.temp.name);(base/'logmel_tables.h').write_bytes(tables)
            subprocess.run(['clang','-O2','-std=c11','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off',
                '-shared','-fPIC','-I'+str(base),str(ROOT/'native/logmel_features.c'),str(ROOT/'native/channel_prepare.c'),
                '-lm','-o',str(base/'logmel.so')],check=True,capture_output=True,timeout=30)
            self.lib=C.CDLL(str(base/'logmel.so'));self.lib.pet_logmel_feature_scratch_size.restype=C.c_size_t
            self.scratch_bytes=self.lib.pet_logmel_feature_scratch_size()
            if not 0<self.scratch_bytes<=32768:raise ValueError('32KiB frontend workspace budget')
            self.scratch=C.create_string_buffer(self.scratch_bytes)
            for name in ('pet_logmel_features','pet_logmel_features_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            for name in ('pet_channel_prepare','pet_channel_prepare_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t]
            self.lib.pet_logmel_pool.argtypes=[C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t]
            self.source_sha256={name:assets.digest(ROOT/name) for name in (*SOURCES,'arcade_ai_pet/logmel_release.py')}
        except BaseException:self.close();raise


class FrozenLogmelModel(assets.DscnnNativeAssets):
    def __init__(self,directory):
        directory=assets.real_path(directory);meta,layers=verify_release(directory)
        tables=assets.real_path(directory/'logmel_tables.h').read_bytes()
        self.frontend=self.engine=None
        try:
            self.frontend=_FrozenFrontend(tables);self.engine=DscnnNativeModel(layers)
            if meta['source_sha256']!=frozen.source_hashes():raise ValueError('frozen source changed during build')
        except BaseException:self.close();raise
        self.layers=layers;self.metadata=meta;self.weights_bytes=meta['weights_bytes']
        self.scratch_bytes=self.frontend.scratch_bytes;self.table=np.array(meta['thresholds_q10'],np.int32)
        self.source_sha256={**self.frontend.source_sha256,**self.engine.source_sha256}
