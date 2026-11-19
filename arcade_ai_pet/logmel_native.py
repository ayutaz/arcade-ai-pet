"""Double-precision host C logmel reference; no model asset or device I/O."""
import ctypes as C
import hashlib
from pathlib import Path
import subprocess,tempfile
import numpy as np
from .features import _WINDOW,_MEL
from .decision_assets import digest

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('native/logmel_features.h','native/logmel_features.c','native/channel_prepare.c',
         'native/channel_features.h','native/pet_kws.h','arcade_ai_pet/logmel_native.py','arcade_ai_pet/features.py')


def render_tables():
    if (not isinstance(_WINDOW,np.ndarray) or _WINDOW.shape!=(400,) or _WINDOW.dtype!=np.float64 or
        not np.isfinite(_WINDOW).all() or not isinstance(_MEL,np.ndarray) or _MEL.shape!=(32,257) or
        _MEL.dtype!=np.float64 or not np.isfinite(_MEL).all() or np.any(_MEL<0) or np.any(_MEL>1)):
        raise ValueError('fixed finite double mathematical tables required')
    indexes=np.full((257,2),-1,np.int8);weights=np.zeros((257,2),np.float64)
    for i in range(257):
        active=np.flatnonzero(_MEL[:,i]>0)
        if len(active)>2:raise ValueError('at most two sparse mel filters per FFT bin')
        indexes[i,:len(active)]=active;weights[i,:len(active)]=_MEL[active,i]
    def array(name,values,ctype):
        def body(row):
            if row.ndim>1:return '{'+','.join(body(r) for r in row)+'}'
            return '{'+','.join(float(v).hex() if ctype=='double' else str(int(v)) for v in row)+'}'
        return 'static const '+ctype+' '+name+''.join(f'[{s}]' for s in values.shape)+' = '+body(values)+';\n'
    text='/* SPDX-License-Identifier: Apache-2.0; generated mathematical tables. */\n'
    for name,values,ctype in (('pet_logmel_window',_WINDOW,'double'),('pet_logmel_index',indexes,'int8_t'),
        ('pet_logmel_weight',weights,'double'),('pet_logmel_cos',np.cos(2*np.pi*np.arange(256)/512),'double'),
        ('pet_logmel_sin',np.sin(2*np.pi*np.arange(256)/512),'double')):text+=array(name,values,ctype)
    return text


class LogmelNativeFrontend:
    def __init__(self):
        tables=render_tables();self.tables_sha256=hashlib.sha256(tables.encode()).hexdigest()
        self.temp=tempfile.TemporaryDirectory();self.closed=False
        try:
            base=Path(self.temp.name);(base/'logmel_tables.h').write_text(tables)
            subprocess.run(['clang','-O2','-std=c11','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off',
                '-shared','-fPIC','-I'+str(base),str(ROOT/'native/logmel_features.c'),str(ROOT/'native/channel_prepare.c'),
                '-lm','-o',str(base/'logmel.so')],check=True,capture_output=True,timeout=30)
            self.lib=C.CDLL(str(base/'logmel.so'));self.lib.pet_logmel_feature_scratch_size.restype=C.c_size_t
            self.scratch_bytes=self.lib.pet_logmel_feature_scratch_size()
            if not 0<self.scratch_bytes<=32768:raise ValueError('32KiB frontend workspace budget')
            self.scratch=C.create_string_buffer(self.scratch_bytes)
            for name in ('pet_logmel_features','pet_logmel_features_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            for name in ('pet_channel_prepare','pet_channel_prepare_banked'):
                getattr(self.lib,name).argtypes=[C.c_void_p,C.c_size_t]
            self.lib.pet_logmel_pool.argtypes=[C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t]
            self.source_sha256={name:digest(ROOT/name) for name in SOURCES}
        except BaseException:self.close();raise

    def close(self):self.closed=True;self.temp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def _storage(self,pcm,banked):
        if (self.closed or not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16 or pcm.shape!=(32000,) or type(banked) is not bool):
            raise ValueError('open frontend, 32000 PCM16 and bool layout required')
        if not banked:
            values=np.array(pcm,copy=True,order='C');return values,None,values.ctypes.data
        banks=[np.zeros(4096,np.int16) for _ in range(8)]
        for i,bank in enumerate(banks):
            chunk=pcm[i*4096:(i+1)*4096];bank[:len(chunk)]=chunk
        pointers=(C.c_void_p*8)(*[b.ctypes.data for b in banks]);return None,banks,pointers

    def _run(self,pcm,banked,prepare,extract):
        values,banks,pointer=self._storage(pcm,banked)
        if prepare:
            fn=self.lib.pet_channel_prepare_banked if banked else self.lib.pet_channel_prepare
            if fn(pointer,32000)!=1:raise ValueError('HP160/level preparation failed')
        if not extract:return values if banks is None else np.concatenate(banks)[:32000]
        out=np.empty(2113,np.float32);fn=self.lib.pet_logmel_features_banked if banked else self.lib.pet_logmel_features
        if fn(pointer,32000,out.ctypes.data,2113,self.scratch,self.scratch_bytes)!=1:raise ValueError('C logmel extraction refused')
        return out

    def prepare(self,pcm,*,banked=False):return self._run(pcm,banked,True,False)
    def features(self,pcm,*,banked=False):return self._run(pcm,banked,False,True)
    def extract(self,pcm,*,banked=False):return self._run(pcm,banked,True,True)

    def pool(self,values):
        if (self.closed or not isinstance(values,np.ndarray) or values.dtype not in (np.dtype('float32'),np.dtype('float64')) or
            values.ndim!=2 or not 1<=len(values)<=198 or not 1<=values.shape[1]<=32 or
            not np.isfinite(values).all() or np.any(abs(values)>1e6)):
            raise ValueError('bounded finite 1..198 by 1..32 floating values required')
        values=np.require(values,dtype=np.float64,requirements=['C','A']);out=np.empty((64,values.shape[1]),np.float64)
        if self.lib.pet_logmel_pool(values.ctypes.data,len(values),values.shape[1],out.ctypes.data,out.size)!=1:
            raise ValueError('C fractional pooling refused')
        return out
