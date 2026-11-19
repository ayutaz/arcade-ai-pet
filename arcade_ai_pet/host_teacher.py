"""Adapters for an explicitly loaded host TTS/ASR pair. No loading or networking."""
import os
from pathlib import Path
import wave
import numpy as np


def source_callbacks(runtime, asr, *, request_factory, resample):
    def synthesize(prompt, caption, seed, path):
        path=Path(path).absolute()
        if path.resolve()!=path or path.exists():
            raise ValueError('fresh real source path required')
        result=runtime.synthesize(request_factory(text=prompt,caption=caption,seed=seed,
                                                 no_ref=True,seconds=2.,num_steps=40))
        values=result.audio.detach().float().cpu().numpy();rate=result.sample_rate
        mono=(values.ndim==1 or values.ndim==2 and values.shape[0]==1 or
              values.ndim==3 and values.shape[:2]==(1,1))
        if (type(rate) is not int or not 8000<=rate<=96000 or not mono or
                not rate*.1<=values.size<=rate*2 or not np.isfinite(values).all()):
            raise ValueError('bounded finite mono synthesis required')
        pcm=np.clip(np.rint(values.reshape(-1)*32767),-32768,32767).astype('<i2')
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as file:
            with wave.open(file,'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate);stream.writeframes(pcm.tobytes())

    def recognize(rate, samples):
        if (type(rate) is not int or not 8000<=rate<=96000 or not isinstance(samples,np.ndarray) or
                samples.dtype!=np.int16 or samples.ndim!=1 or not rate*.1<=len(samples)<=rate*2):
            raise ValueError('bounded PCM16 source required')
        values=samples.astype(np.float32)/32768.
        if rate!=16000:values=resample(values,rate)
        values=np.asarray(values,dtype=np.float32)
        if values.ndim!=1 or not 1600<=len(values)<=32000 or not np.isfinite(values).all():
            raise ValueError('finite resampled source required')
        segments,_=asr.transcribe(np.ascontiguousarray(values),language='ja',beam_size=5,
                                 condition_on_previous_text=False,vad_filter=False,temperature=0.,max_new_tokens=32)
        return ''.join(row.text for row in segments).strip()

    return synthesize,recognize
