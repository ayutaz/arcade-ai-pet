"""A fixed 260-value MFCC/energy front end; no device I/O."""
import hashlib
import numpy as np

RATE=16000
SAMPLES=32000
FRAMES=198
WINDOW=400
HOP=160
FFT=512
MELS=32
COEFFICIENTS=12
BINS=20


def split_for(source_id):
    if type(source_id) is not str or not source_id:raise ValueError('source utterance id required')
    value=int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:4],'little')%100
    return 'test' if value<10 else 'validation' if value<20 else 'train'


def _tables():
    window=np.hamming(WINDOW)
    hz=lambda mel:700*(10**(mel/2595)-1)
    edges=hz(np.linspace(2595*np.log10(1+80/700),2595*np.log10(1+7600/700),MELS+2))
    frequency=np.arange(FFT//2+1)*RATE/FFT
    mel=np.zeros((MELS,len(frequency)))
    for i in range(MELS):
        mel[i]=np.maximum(0,np.minimum((frequency-edges[i])/(edges[i+1]-edges[i]),
                                       (edges[i+2]-frequency)/(edges[i+2]-edges[i+1])))
        mel[i]/=max(mel[i].sum(),1)
    dct=np.cos(np.pi*(np.arange(MELS)+.5)[None,:]*np.arange(1,COEFFICIENTS+1)[:,None]/MELS)*np.sqrt(2/MELS)
    return window,mel,dct


_WINDOW,_MEL,_DCT=_tables()


def features(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(SAMPLES,):
        raise ValueError('exactly 32000 mono PCM16 samples required')
    x=pcm.astype(np.float64);peak=np.max(np.abs(x))
    if peak<32:return np.zeros(260,dtype=np.float32)
    x/=peak
    framed=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
    power=np.abs(np.fft.rfft(framed*_WINDOW,n=FFT))**2
    cepstrum=np.log(np.maximum(power@_MEL.T,1e-8))@_DCT.T
    raw_energy=np.mean(framed**2,axis=1)
    active=np.flatnonzero(raw_energy>=max(.0004,float(raw_energy.max())*.06))
    first,last=(int(active[0]),int(active[-1])+1) if len(active) else (0,FRAMES)
    if last-first<BINS:first=max(0,min(first,FRAMES-BINS));last=first+BINS
    energy=np.log(np.maximum(raw_energy,1e-8));energy-=energy.max()
    groups=np.array_split(np.arange(first,last),BINS)
    mfcc=np.stack([cepstrum[group].mean(axis=0) for group in groups]).ravel()/4
    rms=np.array([energy[group].mean() for group in groups])/2
    return np.clip(np.concatenate((mfcc,rms)),-16,16).astype(np.float32)
