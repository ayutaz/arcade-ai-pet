"""Host-only adaptive MFCC experiment retaining 100 ms on either side."""
import numpy as np
from .features import SAMPLES,WINDOW,HOP,FFT,BINS,_WINDOW,_MEL,_DCT
from .channel_frontend import active_span as legacy_span

CONTEXT_FRAMES=10


def active_span(energy):
    first,last=legacy_span(energy)
    return max(0,first-CONTEXT_FRAMES),min(len(energy),last+CONTEXT_FRAMES)


def features(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(SAMPLES,):
        raise ValueError('exactly 32000 PCM16 samples required')
    x=pcm.astype(np.float64);peak=np.max(abs(x))
    if peak<32:return np.zeros(260,dtype=np.float32)
    x/=peak;framed=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
    energy=np.mean(framed**2,axis=1);first,last=active_span(energy)
    power=abs(np.fft.rfft(framed*_WINDOW,n=FFT))**2
    cepstrum=np.log(np.maximum(power@_MEL.T,1e-8))@_DCT.T
    log_energy=np.log(np.maximum(energy,1e-8));log_energy-=log_energy.max()
    groups=np.array_split(np.arange(first,last),BINS)
    mfcc=np.stack([cepstrum[g].mean(axis=0) for g in groups]).ravel()/4
    rms=np.array([log_energy[g].mean() for g in groups])/2
    return np.clip(np.concatenate((mfcc,rms)),-16,16).astype(np.float32)
