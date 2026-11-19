"""Training-only bounded masks for the existing 64 x 32 logmel image."""
import numpy as np


def validate_plan(plan, rows):
    if (type(rows) is not int or not 1 <= rows <= 1024 or
        not isinstance(plan,np.ndarray) or plan.dtype != np.int32 or plan.shape != (rows,4) or
        np.any(plan < 0) or np.any(plan[:,1] > 6) or np.any(plan[:,3] > 3) or
        np.any(plan[:,0] > 64-plan[:,1]) or np.any(plan[:,2] > 32-plan[:,3])):
        raise ValueError('bounded int32 time-start/width and frequency-start/width plan required')


def make_plan(rows, rng):
    if type(rows) is not int or not 1 <= rows <= 1024 or not isinstance(rng,np.random.Generator):
        raise ValueError('bounded rows and independent NumPy generator required')
    time_width=rng.integers(0,7,size=rows,dtype=np.int32)
    frequency_width=rng.integers(0,4,size=rows,dtype=np.int32)
    plan=np.column_stack((rng.integers(0,65-time_width,size=rows,dtype=np.int32),time_width,
                          rng.integers(0,33-frequency_width,size=rows,dtype=np.int32),frequency_width))
    validate_plan(plan,rows)
    return plan


def apply(values, plan):
    import torch
    if (not isinstance(values,torch.Tensor) or values.dtype != torch.float32 or values.ndim != 2 or
        values.shape[1] != 2113 or values.device.type not in ('cpu','mps') or
        not torch.isfinite(values).all()):
        raise ValueError('finite float32 CPU/MPS batches of 2113 features required')
    validate_plan(plan,len(values))
    bounds=torch.from_numpy(plan).to(values.device)
    time=torch.arange(64,device=values.device)[None,:]
    frequency=torch.arange(32,device=values.device)[None,:]
    times=(time>=bounds[:,0,None]) & (time<bounds[:,0,None]+bounds[:,1,None])
    frequencies=(frequency>=bounds[:,2,None]) & (frequency<bounds[:,2,None]+bounds[:,3,None])
    result=values.clone()
    result[:,:2048].reshape(-1,64,32).masked_fill_(times[:,:,None] | frequencies[:,None,:],0)
    return result
