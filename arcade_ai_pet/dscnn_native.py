"""Trusted host C conformance for the existing six-layer integer CNN, without PCM or deployment."""
import ctypes as C
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .dscnn_model import validate,inputs
from .decision_assets import digest

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('native/dscnn.h','native/dscnn.c','arcade_ai_pet/dscnn_native.py')


class Layer(C.Structure):
    _fields_=[('weight',C.POINTER(C.c_int8)),('bias',C.POINTER(C.c_int32)),('shift',C.POINTER(C.c_int32)),
              ('weight_count',C.c_size_t),('bias_count',C.c_size_t),('shift_count',C.c_size_t)]


class DscnnNativeModel:
    def __init__(self,layers):
        validate(layers)
        self.layers=[{k:np.array(v,copy=True,order='C') for k,v in row.items()} for row in layers]
        self.layer_table=(Layer*6)(*[Layer(row['weight'].ctypes.data_as(C.POINTER(C.c_int8)),
            row['bias'].ctypes.data_as(C.POINTER(C.c_int32)),row['shift'].ctypes.data_as(C.POINTER(C.c_int32)),
            row['weight'].size,row['bias'].size,row['shift'].size) for row in self.layers])
        self.temp=tempfile.TemporaryDirectory();self.closed=False
        try:
            library=Path(self.temp.name)/'dscnn.so'
            subprocess.run(['clang','-O2','-std=c11','-Wall','-Wextra','-Werror','-fno-fast-math','-shared','-fPIC',
                str(ROOT/'native/dscnn.c'),'-lm','-o',str(library)],check=True,capture_output=True,timeout=30)
            self.lib=C.CDLL(str(library));self.lib.pet_dscnn_scratch_size.restype=C.c_size_t
            self.scratch_bytes=self.lib.pet_dscnn_scratch_size()
            if self.scratch_bytes!=12288:raise ValueError('fixed CNN workspace budget changed')
            self.scratch=C.create_string_buffer(self.scratch_bytes)
            self.lib.pet_dscnn_predict.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            self.lib.pet_dscnn_classify.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            self.source_sha256={name:digest(ROOT/name) for name in SOURCES}
        except BaseException:
            self.close();raise

    def close(self):
        self.closed=True;self.temp.cleanup()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def scores(self,values):
        if self.closed:raise ValueError('closed native model')
        inputs(values)
        values=np.require(values,dtype=np.float32,requirements=['C','A'])
        result=np.empty((len(values),7),np.int32)
        for row,out in zip(values,result,strict=True):
            if self.lib.pet_dscnn_predict(row.ctypes.data,2113,self.layer_table,6,out.ctypes.data,7,self.scratch,self.scratch_bytes)!=1:
                raise ValueError('native CNN inference refused')
        return result

    def classify(self,logits,thresholds):
        if self.closed:raise ValueError('closed native model')
        if (not isinstance(logits,np.ndarray) or logits.dtype!=np.int32 or logits.shape!=(7,) or
            not isinstance(thresholds,np.ndarray) or thresholds.dtype!=np.int32 or thresholds.shape!=(5,) or
            np.any(thresholds<0) or np.any(thresholds>4096)):
            raise ValueError('seven int32 logits and five bounded int32 thresholds required')
        logits=np.require(logits,requirements=['C','A']);thresholds=np.require(thresholds,requirements=['C','A'])
        result=self.lib.pet_dscnn_classify(logits.ctypes.data,7,thresholds.ctypes.data,5)
        if not 0<=result<=6:raise ValueError('native CNN classification refused')
        return result
