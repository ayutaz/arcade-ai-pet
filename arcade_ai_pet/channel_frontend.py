"""Experimental adaptive speech window / stationary subtraction, host only."""
import numpy as np
from .features import SAMPLES,FRAMES,WINDOW,HOP,FFT,BINS,_WINDOW,_MEL,_DCT


def active_span(energy):
    if not isinstance(energy,np.ndarray) or energy.shape!=(FRAMES,) or not np.isfinite(energy).all() or (energy<0).any():
        raise ValueError('198 finite nonnegative frame energies required')
    maximum=float(energy.max());baseline=float(np.quantile(energy,.20,method='lower'))
    limit=max(.0004,maximum*.06,min(baseline*3,maximum*.8))
    active=np.flatnonzero(energy>=limit)
    first,last=(int(active[0]),int(active[-1])+1) if len(active) else (0,FRAMES)
    if last-first<BINS:first=max(0,min(first,FRAMES-BINS));last=first+BINS
    return first,last


def subtract_noise(power,energy):
    if (not isinstance(power,np.ndarray) or power.shape!=(FRAMES,257) or not np.isfinite(power).all() or
        (power<0).any()):raise ValueError('finite nonnegative FFT power required')
    active_span(energy)  # Validate before indexing.
    baseline=np.quantile(energy,.20,method='lower')
    floor=power[energy<=baseline].mean(axis=0)
    return np.maximum(power-floor,power*.05)


def features(pcm,*,denoise=False):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(SAMPLES,) or type(denoise) is not bool:
        raise ValueError('exactly 32000 PCM16 samples and bool denoise required')
    x=pcm.astype(np.float64);peak=np.max(abs(x))
    if peak<32:return np.zeros(260,dtype=np.float32)
    x/=peak;framed=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
    energy=np.mean(framed**2,axis=1);first,last=active_span(energy)
    power=abs(np.fft.rfft(framed*_WINDOW,n=FFT))**2
    if denoise:power=subtract_noise(power,energy)
    cepstrum=np.log(np.maximum(power@_MEL.T,1e-8))@_DCT.T
    log_energy=np.log(np.maximum(energy,1e-8));log_energy-=log_energy.max()
    groups=np.array_split(np.arange(first,last),BINS)
    mfcc=np.stack([cepstrum[g].mean(axis=0) for g in groups]).ravel()/4
    rms=np.array([log_energy[g].mean() for g in groups])/2
    return np.clip(np.concatenate((mfcc,rms)),-16,16).astype(np.float32)
