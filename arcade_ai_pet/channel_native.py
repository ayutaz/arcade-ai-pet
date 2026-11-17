"""Validated numeric exports and host C runtime for the frozen adaptive model."""
import ctypes as C
import hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
from .channel_freeze import _verify, _channel_features
from .channel_robustness import PROFILES,simulate,measure
from .native_evaluation import _examples
from .training import scores,decisions
from .frozen import read_layers
from .native import _render_tables

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=dict(conditioning='hp160',frontend='adaptive',peak=4096,samples=32000,rate=16000)


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_channel(frozen,output):
    frozen=Path(frozen).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh channel asset output required')
    meta,layers=_verify(frozen)
    if {k:meta[k] for k in CONTRACT}!=CONTRACT:raise ValueError('unsupported native channel contract')
    tables,model=_render_tables(layers,meta['threshold_q10'])
    output.mkdir(parents=True)
    for name,value in [('feature_tables.h',tables),('pet_model.h',model)]:(output/name).write_text(value)
    shutil.copyfile(frozen/'weights.npz',output/'weights.npz')
    result=dict(runtime='channel-adaptive-v1',input_contract=CONTRACT.copy(),threshold_q10=meta['threshold_q10'],
                weights_sha256=meta['weights_sha256'],weights_bytes=meta['weights_bytes'],
                frozen_sha256=_hash(frozen/'channel-frozen.json'),
                header_sha256={name:_hash(output/name) for name in ('feature_tables.h','pet_model.h')},
                device_tested=False,real_voice_tested=False)
    (output/'channel-metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


class ChannelNativeModel:
    def __init__(self,assets):
        assets=Path(assets).resolve();meta=json.loads((assets/'channel-metadata.json').read_text())
        threshold=meta['threshold_q10'];self.layers,size=read_layers(assets/'weights.npz')
        if (meta['runtime']!='channel-adaptive-v1' or meta['input_contract']!=CONTRACT or
            type(threshold) is not int or not 0<=threshold<=4096 or
            _hash(assets/'weights.npz')!=meta['weights_sha256'] or size!=meta['weights_bytes']):
            raise ValueError('native channel model identity/contract')
        tables,model=_render_tables(self.layers,threshold)
        rendered={'feature_tables.h':tables,'pet_model.h':model}
        for name,value in rendered.items():
            digest=hashlib.sha256(value.encode()).hexdigest()
            if _hash(assets/name)!=digest or meta['header_sha256'].get(name)!=digest:
                raise ValueError('channel header differs from validated numeric export')
        self.temp=tempfile.TemporaryDirectory();self.weights_bytes=size;self.metadata=meta
        try:
            base=Path(self.temp.name)
            for name,value in rendered.items():(base/name).write_text(value)
            sources=('channel_features.c','channel_prepare.c','inference.c')
            subprocess.run(['clang','-O2','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',
                            '-I'+str(ROOT/'native'),'-I'+str(base),
                            *[str(ROOT/'native'/s) for s in sources],'-lm','-o',str(base/'channel.so')],
                           capture_output=True,check=True,timeout=30)
            self.lib=C.CDLL(str(base/'channel.so'))
            self.lib.pet_channel_feature_scratch_size.restype=C.c_size_t
            self.scratch_bytes=self.lib.pet_channel_feature_scratch_size()
            if self.scratch_bytes>24576:raise ValueError('channel scratch budget')
            self.scratch=C.create_string_buffer(self.scratch_bytes)
            for name in ('pet_channel_prepare','pet_channel_prepare_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t]
            for name in ('pet_channel_features','pet_channel_features_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            self.lib.pet_predict.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            self.source_sha256={s:_hash(ROOT/'native'/s) for s in (*sources,'channel_features.h','pet_kws.h')}
        except BaseException:self.temp.cleanup();raise

    def __enter__(self):return self
    def __exit__(self,*args):self.temp.cleanup()

    def _prepared(self,pcm,banked):
        if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,) or type(banked) is not bool:
            raise ValueError('32000 PCM16 samples and bool banked required')
        if banked:
            banks=[np.zeros(4096,dtype=np.int16) for _ in range(8)]
            for i,b in enumerate(banks):
                chunk=pcm[i*4096:(i+1)*4096];b[:len(chunk)]=chunk
            pointer=(C.c_void_p*8)(*[b.ctypes.data for b in banks])
            if self.lib.pet_channel_prepare_banked(pointer,32000)!=1:raise ValueError('banked preparation failed')
            return None,banks,pointer
        values=np.array(pcm,copy=True,order='C')
        if self.lib.pet_channel_prepare(values.ctypes.data,32000)!=1:raise ValueError('preparation failed')
        return values,None,None

    def prepare(self,pcm,*,banked=False):
        values,banks,_=self._prepared(pcm,banked)
        return values if banks is None else np.concatenate(banks)[:32000]

    def extract(self,pcm,*,banked=False):
        values,banks,pointer=self._prepared(pcm,banked);result=np.zeros(260,dtype=np.float32)
        fn=self.lib.pet_channel_features_banked if banked else self.lib.pet_channel_features
        if fn(pointer if banked else values.ctypes.data,32000,result.ctypes.data,260,self.scratch,self.scratch_bytes)!=1:
            raise ValueError('channel features failed')
        return result

    def predict(self,values):
        if not isinstance(values,np.ndarray) or values.dtype!=np.float32 or values.shape!=(260,) or not np.isfinite(values).all():
            raise ValueError('finite 260 float32 features required')
        values=np.ascontiguousarray(values);logits=np.zeros(7,dtype=np.int32)
        label=self.lib.pet_predict(values.ctypes.data,260,logits.ctypes.data,7)
        if not 0<=label<=6:raise ValueError('channel inference failed')
        return label,logits


def evaluate_native_channel(frozen,assets,dataset,python_report,output):
    import time
    frozen=Path(frozen).resolve();assets=Path(assets).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh native audit output required')
    meta,layers=_verify(frozen);reference=json.loads(Path(python_report).read_text())
    if (reference.get('weights_sha256')!=meta['weights_sha256'] or
        reference.get('frozen_sha256')!=_hash(frozen/'channel-frozen.json') or
        reference.get('dataset_manifest_sha256')!=_hash(dataset/'manifest.json')):
        raise ValueError('native audit reference identity mismatch')
    asset_meta=json.loads((assets/'channel-metadata.json').read_text())
    if asset_meta['weights_sha256']!=meta['weights_sha256'] or asset_meta['threshold_q10']!=meta['threshold_q10']:
        raise ValueError('native assets differ from frozen model')
    manifest=json.loads((dataset/'manifest.json').read_text())
    ids={r['id'] for r in manifest['records']};waves={r['sha256'] for r in manifest['records'] if 'sha256' in r}
    if ids & set(meta['source_ids']) or waves & set(meta['wave_hashes']):raise ValueError('audit source/wave overlap')
    expanded=[r['id'] for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
    labels=[];predictions={p:[] for p in PROFILES};exact=matched=banked_equal=0;maximum=0.0;timings=[]
    with ChannelNativeModel(assets) as model:
        for (pcm,label,_),ident in zip(_examples(dataset),expanded,strict=True):
            labels.append(label);seed=int(ident[:16],16)
            for profile in PROFILES:
                expected=_channel_features(pcm,profile,seed,meta)
                python_logits=scores(expected[None,:],layers)[0]
                python_label=int(decisions(python_logits[None,:],meta['threshold_q10'])[0])
                _,direct_logits=model.predict(expected);exact+=int(np.array_equal(direct_logits,python_logits))
                changed=simulate(pcm,profile,seed)
                start=time.perf_counter_ns();actual=model.extract(changed)
                timings.append((time.perf_counter_ns()-start)/1e6)
                banked=model.extract(changed,banked=True)
                banked_equal+=int(np.array_equal(actual,banked));maximum=max(maximum,float(abs(actual-expected).max()))
                native_label,_=model.predict(actual);matched+=int(native_label==python_label)
                predictions[profile].append(native_label)
        total=len(labels)*len(PROFILES)
        rows=[dict(profile=p,**measure(labels,predictions[p])) for p in PROFILES]
        conform=bool(total and exact==total and matched==total and banked_equal==total and maximum<.001)
        quality=all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in rows)
        result=dict(schema=1,scope='host_native_frozen_channel_conformance_and_synthetic_quality',
                    examples_per_profile=len(labels),total_profile_examples=total,profiles=rows,
                    direct_integer_logit_matches=exact,end_to_end_decision_matches=matched,banked_feature_matches=banked_equal,
                    maximum_feature_error=maximum,scratch_bytes=model.scratch_bytes,weights_bytes=model.weights_bytes,
                    weights_sha256=meta['weights_sha256'],threshold_q10=meta['threshold_q10'],
                    frozen_sha256=reference['frozen_sha256'],dataset_manifest_sha256=reference['dataset_manifest_sha256'],
                    source_sha256=model.source_sha256,header_sha256=asset_meta['header_sha256'],
                    conformance_passed=conform,quality_passed=quality,accepted_for_target_build=conform and quality,
                    host_contiguous_extract_median_ms=float(np.median(timings)),host_contiguous_extract_p95_ms=float(np.percentile(timings,95)),
                    host_timings_only=True,target_build_tested=False,device_tested=False,real_voice_tested=False,vast_usd=0)
    output.mkdir(parents=True);(output/'channel-native-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
