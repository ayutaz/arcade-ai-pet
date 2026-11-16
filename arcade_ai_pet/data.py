"""Licensed synthetic sources; split source IDs before any augmentation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave
import numpy as np
from .features import features, split_for

SOURCES={0:['あそぼ'],1:['おいで'],2:['まって'],3:['おやすみ'],4:['ありがとう'],
         5:['こんにちは','さようなら','ゲーム','うたって','すすむ','うしろ','てんき','スタート',
            'とまれ','がんばれ','ごはん','かわいい','はやく','すごい','きょうはあめ','なまえはなに']}


def read_pcm(file):
    with wave.open(str(file),'rb') as stream:
        if stream.getnchannels()!=1 or stream.getsampwidth()!=2 or stream.getcomptype()!='NONE':
            raise ValueError('mono PCM16 WAV required')
        rate=stream.getframerate();pcm=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2')
    if not 8000<=rate<=96000 or not 1<=len(pcm)<=rate*10:raise ValueError('WAV duration/rate')
    return np.rint(np.interp(np.arange(round(len(pcm)*16000/rate))*rate/16000,
                            np.arange(len(pcm)),pcm)).astype(np.int16)


def augment(pcm,seed):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.ndim!=1 or not 1<=len(pcm)<=32000:
        raise ValueError('at most two seconds of PCM16 required')
    rng=np.random.default_rng(seed);x=pcm.astype(np.float64)
    # Add a small, deterministic room response, gain, offset, noise and placement.
    delay=int(rng.integers(80,801));echo=np.zeros_like(x)
    echo[delay:]=x[:-delay]*rng.uniform(0,.25)
    x=(x+echo)*rng.uniform(.15,1.1)
    out=np.zeros(32000);start=int(rng.integers(0,32000-len(x)+1));out[start:start+len(x)]=x
    out+=rng.normal(0,rng.uniform(0,120),32000)
    return np.clip(np.rint(out),-32767,32767).astype(np.int16)


def trim_silence(pcm):
    x=pcm.astype(np.int32);peak=int(np.max(np.abs(x)))
    active=np.flatnonzero(np.abs(x)>=max(32,peak*.015))
    if not len(active):raise ValueError('source has no speech')
    result=pcm[max(0,int(active[0])-800):min(len(pcm),int(active[-1])+801)]
    if len(result)>32000:raise ValueError('speech exceeds recording window; do not truncate')
    return result


def create(output,jtalk_root,*,variants=3,source_cache=None):
    output=Path(output).resolve();root=Path(jtalk_root).resolve()
    if output.exists():raise ValueError('fresh dataset directory required')
    if type(variants) is not int or not 1<=variants<=10:raise ValueError('augmentation count')
    voices=sorted((root/'voice').glob('*/*.htsvoice'))
    if not voices or not (root/'dic/sys.dic').is_file():raise ValueError('Open JTalk dictionary/voices required')
    fingerprint=dict(engine_sha256=hashlib.sha256((root/'bin/open_jtalk').read_bytes()).hexdigest(),
                  voice_sha256={v.name:hashlib.sha256(v.read_bytes()).hexdigest() for v in voices},
                  dictionary_sha256=hashlib.sha256((root/'dic/sys.dic').read_bytes()).hexdigest())
    cached={};cache=None
    if source_cache is not None:
        cache=Path(source_cache).resolve();metadata=json.loads((cache/'manifest.json').read_text())
        if any(metadata.get(name)!=value for name,value in fingerprint.items()):raise ValueError('source engine mismatch')
        cached={row['id']:row['sha256'] for row in metadata['records'] if 'sha256' in row}
    output.mkdir(parents=True);(output/'sources').mkdir()
    records=[];arrays={split:[] for split in ('train','validation','test')}
    labels={split:[] for split in arrays}
    for voice in voices:
        for speed in (.85,1.1,1.35):
            for pitch in (-2,2):
                for label,texts in SOURCES.items():
                    for text in texts:
                        source=f'{voice.parent.name}/{voice.name}/{speed}/{pitch}/{text}'
                        ident=hashlib.sha256(source.encode()).hexdigest();split=split_for(source)
                        file=output/'sources'/(ident+'.wav')
                        if cache is not None:
                            source_file=cache/'sources'/(ident+'.wav')
                            if hashlib.sha256(source_file.read_bytes()).hexdigest()!=cached.get(ident):
                                raise ValueError('cached utterance hash mismatch')
                            shutil.copyfile(source_file,file)
                        else:
                            subprocess.run([str(root/'bin/open_jtalk'),'-x',str(root/'dic'),'-m',str(voice),
                                '-r',str(speed),'-fm',str(pitch),'-ow',str(file)],input=text.encode(),
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=True)
                        pcm=trim_silence(read_pcm(file))
                        for variant in range(variants):
                            seed=int(ident[:8],16)+variant
                            arrays[split].append(features(augment(pcm,seed)));labels[split].append(label)
                        records.append(dict(id=ident,source=source,split=split,label=label,
                            sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    # Noise-only examples have their own independent source IDs and seeds.
    for index in range(200):
        source=f'noise/{index}';split=split_for(source)
        rng=np.random.default_rng(index+70000)
        pcm=np.clip(rng.normal(0,0 if index%4==0 else rng.uniform(1,200),32000),-32767,32767).astype(np.int16)
        arrays[split].append(features(pcm));labels[split].append(6)
        records.append(dict(id=hashlib.sha256(source.encode()).hexdigest(),source=source,split=split,label=6))
    for split in arrays:
        np.savez_compressed(output/(split+'.npz'),x=np.array(arrays[split],dtype=np.float32),y=np.array(labels[split],dtype=np.int64))
    manifest=dict(engine='Open JTalk Homebrew build; executable version must be recorded separately',
                  **fingerprint,feature_version='aligned-mfcc-v2',
                  license='CC-BY-3.0 voices; see docs/data-license.md',variants=variants,records=records)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return {split:len(arrays[split]) for split in arrays}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--jtalk-root',required=True)
    parser.add_argument('--variants',type=int,default=3);args=parser.parse_args()
    print(json.dumps(create(args.output,args.jtalk_root,variants=args.variants)))
