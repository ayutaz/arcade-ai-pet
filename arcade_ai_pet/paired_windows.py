"""Pure host diagnostics for paired conditioned PCM; no model, files or device I/O."""
import numpy as np
from .features import SAMPLES,FRAMES,WINDOW,HOP,FFT,_WINDOW,_MEL
from .channel_frontend import active_span
from .logmel_frontend import features,pool


def _pcm(pcm):
    if not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(SAMPLES,):
        raise ValueError('exactly 32000 mono PCM16 samples required')


def trace(pcm):
    _pcm(pcm)
    values=pcm.astype(np.float64);peak=int(abs(values).max())
    result=dict(peak_amplitude=peak,features_skipped=peak<32,active_span=None,
                context_span=None,context_clipped_frames=None,selected_samples=None)
    if peak<32:return result
    frames=np.lib.stride_tricks.sliding_window_view(values/peak,WINDOW)[::HOP]
    first,last=active_span(np.mean(frames**2,axis=1))
    left,right=max(0,first-10),min(FRAMES,last+10)
    result.update(active_span=[first,last],context_span=[left,right],
                  context_clipped_frames=[max(0,10-first),max(0,last+10-FRAMES)],
                  selected_samples=[left*HOP,(right-1)*HOP+WINDOW])
    return result


def for_span(pcm,*,first,last):
    """Counterfactual existing log-mel features for an explicit exclusive VAD span."""
    _pcm(pcm)
    if type(first) is not int or type(last) is not int or not 0<=first<last<=FRAMES:
        raise ValueError('bounded exclusive integer frame span required')
    x=pcm.astype(np.float64);peak=float(abs(x).max())
    if peak<32:return np.zeros(2113,dtype=np.float32)
    x/=peak;frames=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
    energy=np.mean(frames**2,axis=1)
    left,right=max(0,first-10),min(FRAMES,last+10)
    power=abs(np.fft.rfft(frames*_WINDOW,n=FFT))**2
    bands=pool(np.log(np.maximum(power@_MEL.T,1e-8))[left:right])
    bands=(bands-bands.mean(axis=0))/4
    log_energy=np.log(np.maximum(energy,1e-8));log_energy-=log_energy.max()
    energies=pool(log_energy[left:right,None]).ravel()/2
    return np.clip(np.concatenate((bands.ravel(),energies,[4*(last-first)/FRAMES])),-16,16).astype(np.float32)


def _rmse(a,b):
    return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))


def pair(reference,changed,*,shift_frames=0):
    """Measure a declared 0/±150ms profile shift; a clean span is not ground truth."""
    _pcm(reference);_pcm(changed)
    if type(shift_frames) is not int or shift_frames not in (-15,0,15):
        raise ValueError('declared clean/hum/noise/early/late frame shift required')
    a,b=trace(reference),trace(changed)
    ax,bx=features(reference),features(changed)
    result=dict(scope='paired_conditioned_pcm_window_diagnostic',reference=a,changed=b,
                shift_frames=shift_frames,aligned_reference_span=None,
                aligned_reference_clipped_frames=None,active_boundary_l1_frames=None,
                logmel_rmse=_rmse(ax[:2048],bx[:2048]),energy_rmse=_rmse(ax[2048:2112],bx[2048:2112]),
                duration_delta=float(np.float64(bx[-1])-np.float64(ax[-1])),
                counterfactual_to_reference_rmse=None,counterfactual_to_changed_rmse=None)
    if a['features_skipped']:return result
    first,last=a['active_span'];left,right=max(0,first+shift_frames),min(FRAMES,last+shift_frames)
    result.update(aligned_reference_span=[left,right],aligned_reference_clipped_frames=(last-first)-(right-left))
    if not b['features_skipped']:
        result['active_boundary_l1_frames']=abs(left-b['active_span'][0])+abs(right-b['active_span'][1])
    counterfactual=for_span(changed,first=left,last=right)
    result.update(counterfactual_to_reference_rmse=_rmse(ax,counterfactual),
                  counterfactual_to_changed_rmse=_rmse(bx,counterfactual))
    return result
