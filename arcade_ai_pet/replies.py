"""Five pre-synthesized replies, 8kHz PCM8 assets; not on-device free-form TTS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .data import read_pcm,trim_silence
from .native import _array

REPLIES=('いっしょにあそぼ','はーい','わかった','おやすみ','ありがとう')


def encode_reply(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.ndim!=1 or not 160<=len(pcm)<=32000:
        raise ValueError('bounded PCM16 reply')
    x=pcm.astype(np.float64);x-=x.mean()
    if np.max(abs(x))<32:raise ValueError('silent reply')
    t=np.arange(-15,16);kernel=.45*np.sinc(.45*t)*np.hamming(31);kernel/=kernel.sum()
    x=np.convolve(x,kernel,mode='same')[::2];peak=np.max(abs(x))
    x*=100/max(1,peak)
    fade=np.minimum(np.minimum(np.arange(len(x)),np.arange(len(x))[::-1]),40)/40
    return np.clip(np.rint(x*fade),-100,100).astype(np.int8)


def decode_reply(encoded):
    if not isinstance(encoded,np.ndarray) or encoded.dtype!=np.int8 or encoded.ndim!=1 or not 80<=len(encoded)<=16000:
        raise ValueError('bounded PCM8 reply')
    return np.repeat(encoded.astype(np.int16)*28,2)


def export_replies(clips,output):
    output=Path(output)
    if output.exists() or len(clips)!=5:raise ValueError('five replies and fresh output')
    for clip in clips:decode_reply(clip)
    size=sum(len(x) for x in clips)
    if size>80000:raise ValueError('reply storage budget')
    text='''/* SPDX-License-Identifier: CC-BY-3.0
 * Synthesized NIT ATR503 M001 speech, HTS Working Group.
 * (c)2003-2012 Nagoya Institute of Technology, (c)2003-2008 Tokyo Institute of Technology.
 * Generated, DC removed, low-pass/8kHz resampled, PCM8 quantized and faded.
 * Attribution/license/provenance: assets/replies-v1/model-card.md and LICENSE.txt.
 * Pre-generated clips, not real-time TTS. No device or human recording.
 */
#include <stdint.h>
typedef struct {const int8_t *samples;uint32_t frames;} PetReplyClip;
'''
    for i,clip in enumerate(clips):text+=_array(f'pet_reply_{i}',clip,'int8_t')
    text+='static const PetReplyClip pet_replies[5]={'+','.join(f'{{pet_reply_{i},{len(c)}}}' for i,c in enumerate(clips))+'};\n'
    output.mkdir(parents=True);(output/'replies.h').write_text(text)
    report=dict(samples_hz=8000,encoding='signed PCM8, repeat twice at 16kHz, multiply by 28',encoded_bytes=size,
                frames=[len(c) for c in clips],texts=list(REPLIES),asset_sha256=hashlib.sha256(text.encode()).hexdigest(),
                on_device_tts=False,device_tested=False,license='CC-BY-3.0')
    (output/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report


def synthesize(root,output):
    root=Path(root);output=Path(output)
    engine=root/'bin/open_jtalk';dictionary=root/'dic/sys.dic';voice=root/'voice/m100/nitech_jp_atr503_m001.htsvoice'
    if output.exists() or not all(p.is_file() for p in (engine,dictionary,voice)):raise ValueError('engine/dictionary/voice or output')
    clips=[];hashes=[]
    with tempfile.TemporaryDirectory() as temporary:
        for i,text in enumerate(REPLIES):
            wave=Path(temporary)/f'{i}.wav'
            subprocess.run([str(engine),'-x',str(dictionary.parent),'-m',str(voice),'-r','1.1','-ow',str(wave)],
                           input=text.encode(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=True)
            pcm=trim_silence(read_pcm(wave));hashes.append(hashlib.sha256(pcm.tobytes()).hexdigest());clips.append(encode_reply(pcm))
    report=export_replies(clips,output)
    report.update(engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),dictionary_sha256=hashlib.sha256(dictionary.read_bytes()).hexdigest(),
                  voice_sha256=hashlib.sha256(voice.read_bytes()).hexdigest(),source_pcm_sha256=hashes,speed=1.1)
    (output/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jtalk-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(synthesize(args.jtalk_root,args.output)))
