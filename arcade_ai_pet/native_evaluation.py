"""Reconstruct licensed synthetic inputs and audit the full native C path."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
from .data import read_pcm,trim_silence,augment
from .features import features
from .frozen import read_layers
from .native import export_assets
from .training import load_dataset,scores,decisions,metrics

ROOT=Path(__file__).resolve().parents[1]


class NativeModel:
    def __init__(self,assets):
        assets=Path(assets).resolve();metadata=json.loads((assets/'metadata.json').read_text())
        digest=hashlib.sha256((assets/'weights.npz').read_bytes()).hexdigest()
        threshold=metadata['threshold_q10']
        if (digest!=metadata['weights_sha256'] or type(threshold) is not int or not 0<=threshold<=4096 or
            metadata['feature_version']!='aligned-mfcc-v2'):raise ValueError('native model identity/threshold')
        self.layers,self.weights_bytes=read_layers(assets/'weights.npz')
        self.threshold=threshold;self.weights_sha256=digest
        self.temp=tempfile.TemporaryDirectory()
        try:
            base=Path(self.temp.name);raw=base/'frozen';raw.mkdir()
            shutil.copyfile(assets/'weights.npz',raw/'weights.npz')
            (raw/'frozen.json').write_text(json.dumps(dict(weights_sha256=digest,threshold_q10=threshold,feature_version='aligned-mfcc-v2')))
            # Regenerate numeric-only headers; never compile an arbitrary input header.
            safe=export_assets(raw,base/'safe')
            for name,digest_header in safe['asset_sha256'].items():
                if hashlib.sha256((assets/name).read_bytes()).hexdigest()!=digest_header:
                    raise ValueError('model header differs from validated numeric export')
            library=base/'model.dylib'
            command=['clang','-std=c11','-O2','-shared','-fPIC','-I'+str(ROOT/'native'),'-I'+str(base/'safe'),
                     str(ROOT/'native/features.c'),str(ROOT/'native/inference.c'),'-lm','-o',str(library)]
            subprocess.run(command,capture_output=True,check=True,timeout=30)
            self.lib=ctypes.CDLL(str(library));self.lib.pet_feature_scratch_size.restype=ctypes.c_size_t
            self.lib.pet_features.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
            self.lib.pet_predict.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
            self.scratch_bytes=self.lib.pet_feature_scratch_size();self.scratch=ctypes.create_string_buffer(self.scratch_bytes)
            self.source_sha256={name:hashlib.sha256((ROOT/'native'/name).read_bytes()).hexdigest()
                                for name in ('pet_kws.h','features.c','inference.c')}
        except BaseException:self.temp.cleanup();raise

    def __enter__(self):return self
    def __exit__(self,*args):self.temp.cleanup()

    def extract(self,pcm):
        if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,):raise ValueError('PCM16 input')
        pcm=np.ascontiguousarray(pcm);result=np.zeros(260,dtype=np.float32)
        if self.lib.pet_features(pcm.ctypes.data,32000,result.ctypes.data,260,self.scratch,self.scratch_bytes)!=1:
            raise ValueError('native feature extraction failed')
        return result

    def predict(self,x):
        if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(260,) or not np.isfinite(x).all():
            raise ValueError('finite float32 features required')
        x=np.ascontiguousarray(x);logits=np.zeros(7,dtype=np.int32)
        result=self.lib.pet_predict(x.ctypes.data,260,logits.ctypes.data,7)
        if not 0<=result<=6:raise ValueError('native integer inference failed')
        return result,logits


def _examples(dataset):
    metadata=json.loads((dataset/'manifest.json').read_text());data=load_dataset(dataset)
    positions={split:0 for split in data}
    for row in metadata['records']:
        if 'sha256' in row:
            file=dataset/'sources'/(row['id']+'.wav')
            if hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('synthetic voice source hash')
            raw=trim_silence(read_pcm(file));pcms=[augment(raw,int(row['id'][:8],16)+i) for i in range(metadata['variants'])]
        else:
            if row['label']!=6 or not re.fullmatch('[a-z][a-z0-9-]{0,31}/[0-9]{1,3}',row['source']):
                raise ValueError('synthetic noise source contract')
            namespace,index=row['source'].split('/');index=int(index)
            if not 0<=index<200:raise ValueError('noise index')
            seed=index+70000 if namespace=='noise' else int.from_bytes(hashlib.sha256(row['source'].encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            pcm=np.clip(rng.normal(0,0 if index%4==0 else rng.uniform(1,200),32000),-32767,32767).astype(np.int16)
            if 'pcm_sha256' in row and hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']:
                raise ValueError('noise PCM source hash')
            pcms=[pcm]
        for pcm in pcms:
            split=row['split'];reference=data[split]['x'][positions[split]];positions[split]+=1
            if float(abs(features(pcm)-reference).max())>1e-6:raise ValueError('synthetic reconstruction differs from saved features')
            yield pcm,row['label'],reference


def evaluate_native(assets,dataset,output):
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh native evaluation output required')
    actual=[];expected=[];labels=[];latencies=[];exact=0;maximum=0.0
    with NativeModel(assets) as model:
        for pcm,label,reference in _examples(dataset):
            raw=scores(reference[None,:],model.layers)[0]
            direct,direct_logits=model.predict(reference)
            exact+=int(np.array_equal(raw,direct_logits))
            start=time.perf_counter_ns();native_features=model.extract(pcm);result,_=model.predict(native_features)
            latencies.append((time.perf_counter_ns()-start)/1000000)
            maximum=max(maximum,float(abs(native_features-reference).max()))
            actual.append(result);expected.append(int(decisions(raw[None,:],model.threshold)[0]));labels.append(label)
        count=len(actual)
        report=dict(examples=count,exact_integer_score_matches=exact,end_to_end_decision_matches=sum(a==b for a,b in zip(actual,expected,strict=True)),
            maximum_feature_error=maximum,scratch_bytes=model.scratch_bytes,weights_bytes=model.weights_bytes,
            weights_sha256=model.weights_sha256,source_sha256=model.source_sha256,
            feature_median_host_ms=float(np.median(latencies)),feature_p95_host_ms=float(np.percentile(latencies,95)),
            host_timings_only=True,real_voice_tested=False,device_tested=False,native_metrics=metrics(labels,actual),
            dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest())
        report['native_matches_python']=bool(count and exact==count and actual==expected and maximum<.01 and model.scratch_bytes<=24576)
    output.mkdir(parents=True);(output/'native-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','dataset','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=evaluate_native(args.model,args.dataset,args.output)
    print(json.dumps(result));raise SystemExit(0 if result['native_matches_python'] else 1)
