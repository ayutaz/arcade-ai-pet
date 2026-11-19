"""Keep the exact log-mel prefix and append its removed absolute band means."""
import numpy as np
from . import logmel_frontend as previous
from .features import WINDOW,HOP,FFT,FRAMES,_WINDOW,_MEL
from .channel_frontend import active_span

FEATURE_COUNT=2145


def features(pcm):
    prefix=previous.features(pcm)  # Includes the strict PCM16 and silence contract.
    x=pcm.astype(np.float64);peak=float(abs(x).max())
    if peak<32:return np.zeros(FEATURE_COUNT,dtype=np.float32)
    x/=peak
    framed=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
    first,last=active_span(np.mean(framed**2,axis=1))
    first,last=max(0,first-10),min(FRAMES,last+10)
    logmel=np.log(np.maximum(abs(np.fft.rfft(framed*_WINDOW,n=FFT))**2@_MEL.T,1e-8))
    means=np.clip(previous.pool(logmel[first:last]).mean(axis=0)/4,-16,16).astype(np.float32)
    return np.concatenate((prefix,means))
