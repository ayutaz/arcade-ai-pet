"""Read-only diagnostic stages for known synthetic PCM; no ASR/model loading."""
import numpy as np
from .data import trim_silence,augment
from .channel_robustness import condition
from .channel_training import level
from .channel_frontend import active_span
from .features import WINDOW,HOP,FRAMES


def normalize_for_asr(pcm):
    if (not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.ndim!=1 or
        not 1<=len(pcm)<=192000):
        raise ValueError('bounded mono PCM16 audit input required')
    values=pcm.astype(np.float64);peak=float(abs(values).max())
    if not peak:return pcm.copy()
    return np.clip(np.rint(values*(16384/peak)),-32768,32767).astype(np.int16)


def prepare(pcm,*,seed):
    if (not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.ndim!=1 or
        not 1600<=len(pcm)<=32000 or not np.any(abs(pcm.astype(np.int32))>=32) or
        type(seed) is not int or not 0<=seed<2**32):
        raise ValueError('bounded resampled speech and original uint32 seed required')
    trimmed=trim_silence(pcm)
    rows=[dict(stage='resampled',variant=None,pcm=pcm.copy()),
          dict(stage='trimmed',variant=None,pcm=trimmed.copy())]
    for variant in range(3):
        augmented=augment(trimmed,seed+variant)
        conditioned=level(condition(augmented,'hp160'))
        x=conditioned.astype(np.float64);peak=float(abs(x).max())
        if peak:x/=peak
        frames=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
        first,last=active_span(np.mean(frames**2,axis=1))
        left,right=max(0,first-10),min(FRAMES,last+10)
        start,end=left*HOP,(right-1)*HOP+WINDOW
        rows.extend((dict(stage='augmented',variant=variant,pcm=augmented),
                     dict(stage='conditioned',variant=variant,pcm=conditioned),
                     dict(stage='cropped',variant=variant,pcm=conditioned[start:end].copy(),
                          active_first_frame=first,active_last_frame=last,start_sample=start,end_sample=end)))
    return rows


def padded_context(pcm,*,first,last,margin):
    """Keep one declared context in place; no amplitude or sample-rate changes."""
    if (not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,) or
        type(first) is not int or type(last) is not int or not 0<=first<last<=FRAMES or
        type(margin) is not int or margin not in (10,20)):
        raise ValueError('full PCM16 and bounded declared frame context required')
    left,right=max(0,first-margin),min(FRAMES,last+margin)
    start,end=left*HOP,(right-1)*HOP+WINDOW
    out=np.zeros_like(pcm);out[start:end]=pcm[start:end]
    return dict(pcm=out,start_sample=start,end_sample=end)
