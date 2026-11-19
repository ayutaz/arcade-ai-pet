"""Host-only adaptive MFCC plus explicit active-span duration; 261 values."""
import numpy as np
from .channel_frontend import features as previous, active_span
from .features import FRAMES, WINDOW, HOP


def features(pcm):
    old=previous(pcm)  # Validate the same PCM contract before any new processing.
    x=pcm.astype(np.float64);peak=float(abs(x).max())
    if peak<32:
        return np.zeros(261,dtype=np.float32)
    x/=peak
    energy=np.mean(np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]**2,axis=1)
    first,last=active_span(energy)
    return np.concatenate((old,np.array([4*(last-first)/FRAMES],dtype=np.float32)))
