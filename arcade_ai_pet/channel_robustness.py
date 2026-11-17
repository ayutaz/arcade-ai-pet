"""Fixed synthetic channel stress tests, not a model of measured device PCM."""
import atexit
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .training import metrics
from .native_evaluation import NativeModel, _examples

# Freeze before evaluation; source splitting precedes all channel changes.
PROFILES=('clean','quiet','dc','drift','hum','noise','early','late')
CANDIDATES=('center','hp20','hp80','hp160')


def _pcm(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,):
        raise ValueError('exactly 32000 PCM16 samples required')


def simulate(pcm,profile,seed):
    _pcm(pcm)
    if profile not in PROFILES or type(seed) is not int or not 0<=seed<2**64:raise ValueError('profile/seed')
    x=pcm.astype(np.float64);t=np.arange(32000)/16000
    if profile=='quiet':x*=.02
    elif profile=='dc':x=x*.05+1680
    elif profile=='drift':x=x*.02+1600+480*np.exp(-t/.3)
    elif profile=='hum':x=x*.05+1600+240*np.sin(2*np.pi*50*t)
    elif profile=='noise':x=x*.05+1600+np.random.default_rng(seed).normal(0,30,32000)
    elif profile in ('early','late'):
        # Deliberate truncation stress: not a claim that the recording has this error.
        shift=2400 if profile=='late' else -2400
        original=x.copy();x[:]=0
        if shift>0:x[shift:]=original[:-shift]
        else:x[:shift]=original[-shift:]
    return np.clip(np.rint(x),-32768,32767).astype(np.int16)


@lru_cache(maxsize=1)
def _filter():
    # Trusted repo source only; no arbitrary input C/header is compiled.
    temp=tempfile.TemporaryDirectory();atexit.register(temp.cleanup)
    library=Path(temp.name)/'conditioning.so'
    source=Path(__file__).resolve().parents[1]/'native/conditioning.c'
    subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    str(source),'-lm','-o',str(library)],capture_output=True,check=True,timeout=30)
    lib=ctypes.CDLL(str(library));fn=lib.pet_condition
    fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];fn.restype=ctypes.c_int
    return lib,fn


def condition(pcm,candidate):
    _pcm(pcm)
    if candidate not in ('raw',*CANDIDATES):raise ValueError('conditioning candidate')
    out=np.array(pcm,copy=True,order='C')
    if candidate=='raw':return out
    hz=0 if candidate=='center' else int(candidate[2:])
    if _filter()[1](out.ctypes.data,32000,hz)!=1:raise ValueError('conditioning failed')
    return out


def measure(labels,predictions):
    if (not len(labels) or len(labels)!=len(predictions) or
        any(type(v) not in (int,np.int64,np.int32) or not 0<=v<7 for v in [*labels,*predictions])):
        raise ValueError('nonempty native labels 0..6 required')
    report=metrics(labels,predictions)
    commands=sum(v<5 for v in labels)
    report.update(examples=len(labels),command_recall=sum(a==p for a,p in zip(labels,predictions,strict=True) if a<5)/max(1,commands))
    return report


def check_disjoint(development,heldout):
    if not development or not heldout or development & heldout:raise ValueError('source corpus overlap/empty')
    return True


def choose(rows):
    expected={(c,p) for c in CANDIDATES for p in PROFILES};actual=set()
    for row in rows:
        key=(row['candidate'],row['profile'])
        if key in actual or key not in expected:raise ValueError('duplicate/unknown result')
        actual.add(key)
        for key in ('macro_f1','false_activation_rate','command_recall'):
            value=row[key]
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('metric bounds')
    if actual!=expected:raise ValueError('missing candidate/profile')
    eligible=[]
    for index,candidate in enumerate(CANDIDATES):
        group=[r for r in rows if r['candidate']==candidate]
        if all(r['macro_f1']>=.90 and r['false_activation_rate']<=.05 and r['command_recall']>=.80 for r in group):
            eligible.append((-min(r['macro_f1'] for r in group),-sum(r['macro_f1'] for r in group),index,candidate))
    return min(eligible)[-1] if eligible else None


def compare(examples,predict,*,candidates=CANDIDATES):
    if not examples or not candidates or any(c not in CANDIDATES for c in candidates):raise ValueError('examples/candidates')
    labels=[int(label) for _,label in examples];rows=[]
    for candidate in candidates:
        for profile in PROFILES:
            predictions=[]
            for index,(pcm,_) in enumerate(examples):
                changed=simulate(pcm,profile,index+41000)
                predictions.append(predict(condition(changed,candidate)))
            rows.append(dict(candidate=candidate,profile=profile,**measure(labels,predictions)))
    return rows


def evaluate_channels(assets,development,heldout,output):
    """Select on development validation; open final PCM only after selection."""
    import hashlib,json,time
    development=Path(development).resolve();heldout=Path(heldout).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh evaluation output required')
    manifests=[json.loads((p/'manifest.json').read_text()) for p in (development,heldout)]
    source_sets=[{r['source'] for r in m['records']} for m in manifests]
    check_disjoint(*source_sets)
    wave_hashes=[{r['sha256'] for r in m['records'] if 'sha256' in r} for m in manifests]
    if wave_hashes[0] & wave_hashes[1]:raise ValueError('development/final waveform overlap')
    def collect(path,manifest,split=None):
        splits=[r['split'] for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
        return [(pcm,label) for (pcm,label,_),s in zip(_examples(path),splits,strict=True) if split is None or s==split]
    start=time.perf_counter()
    with NativeModel(assets) as model:
        def predict(pcm):return model.predict(model.extract(pcm))[0]
        validation=compare(collect(development,manifests[0],'validation'),predict)
        selected=choose(validation);final=[]
        if selected is not None:
            final=compare(collect(heldout,manifests[1]),predict,candidates=(selected,))
        report=dict(schema=1,scope='synthetic_channel_stress_not_measured_device',
                    selected_candidate=selected,final_evaluated=selected is not None,
                    validation=validation,final=final,
                    selection_gates=dict(each_profile_macro_f1=.90,max_false_activation_rate=.05,min_command_recall=.80),
                    profiles=list(PROFILES),candidates=list(CANDIDATES),weights_sha256=model.weights_sha256,
                    dataset_manifest_sha256={name:hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()
                                             for name,p in [('development',development),('final',heldout)]},
                    source_overlap=0,device_tested=False,real_voice_tested=False,vast_usd=0,
                    host_elapsed_seconds=time.perf_counter()-start)
    output.mkdir(parents=True);(output/'channel-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','development','heldout','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    report=evaluate_channels(args.model,args.development,args.heldout,args.output)
    print(json.dumps({k:report[k] for k in ('selected_candidate','final_evaluated','host_elapsed_seconds')}))
