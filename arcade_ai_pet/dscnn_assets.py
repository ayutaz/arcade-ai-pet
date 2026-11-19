"""Strict numeric assets for the fixed 2113-input logmel CNN, not a firmware image."""
import hashlib,json,math,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
from .decision_assets import real_path,digest,thresholds
from .dscnn_model import SHAPES,validate,weights_bytes
from .native import _array
from .logmel_native import LogmelNativeFrontend,render_tables
from .dscnn_native import DscnnNativeModel

RUNTIME='context-logmel-dscnn-v1'
CONTRACT=dict(conditioning='hp160',frontend='context-logmel64',peak=4096,samples=32000,rate=16000,
              features=2113,context_frames=10,time_bins=64,mel_bands=32,duration_scale=4,
              pooling='fractional_frame_overlap',band_normalization='per_band_mean_divide4',
              activation='signed_Q5_then_uint8_Q5_ReLU6',logits='signed_int32_Q10')


def read_layers(path):
    path=real_path(path)
    if not path.is_file() or path.stat().st_size>262144:raise ValueError('bounded regular numeric archive required')
    expected={f'{key}{i}.npy':(np.dtype('int8') if key=='weight' else np.dtype('int32'),shape if key=='weight' else (shape[0],))
              for i,shape in enumerate(SHAPES) for key in ('weight','bias','shift')}
    try:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist())!=18 or set(archive.namelist())!=set(expected):raise ValueError('exactly eighteen fixed numeric arrays required')
            for info in archive.infolist():
                if info.file_size>32768:raise ValueError('numeric array budget')
                with archive.open(info) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=2048)
                    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=2048)
                    else:raise ValueError('supported numeric format required')
                    if (fortran or (dtype,shape)!=expected[info.filename] or
                        stream.tell()+math.prod(shape)*dtype.itemsize!=info.file_size):raise ValueError('fixed typed architecture and payload required')
        with np.load(path,allow_pickle=False) as arrays:layers=[{key:arrays[f'{key}{i}'] for key in ('weight','bias','shift')} for i in range(6)]
    except (zipfile.BadZipFile,EOFError,OSError) as error:raise ValueError('readable bounded numeric archive required') from error
    validate(layers);size=weights_bytes(layers)
    if size+20>24576:raise ValueError('24KiB numeric budget')
    return layers,size


def render(layers,table):
    validate(layers);table=thresholds(table)
    text='/* Generated numeric model; model/data licensing is separate from code. */\n#include "dscnn.h"\n'
    for i,row in enumerate(layers):
        for key in ('weight','bias','shift'):text+=_array(f'pet_logmel_{key}{i}',row[key],'int8_t' if key=='weight' else 'int32_t')
    text+=_array('pet_logmel_thresholds',np.array(table,np.int32),'int32_t')
    text+='static const PetDscnnLayer pet_logmel_layers[6] = {\n'
    for i,row in enumerate(layers):
        text+='{(const int8_t *)pet_logmel_weight'+str(i)+',pet_logmel_bias'+str(i)+',pet_logmel_shift'+str(i)+','+str(row['weight'].size)+','+str(row['bias'].size)+','+str(row['shift'].size)+'},\n'
    text+='};\n'
    return {'logmel_tables.h':render_tables(),'logmel_model.h':text}


def export_assets(weights,table,output):
    weights=real_path(weights);output=real_path(output);table=thresholds(table)
    if output.exists():raise ValueError('fresh numeric output required')
    before=digest(weights);layers,size=read_layers(weights);headers=render(layers,table)
    if digest(weights)!=before:raise ValueError('numeric input changed')
    meta=dict(runtime=RUNTIME,input_contract=CONTRACT.copy(),thresholds_q10=table,weights_sha256=before,
              weights_bytes=size,numeric_bytes=size+20,header_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in headers.items()},
              frozen=False,device_tested=False,real_voice_tested=False)
    output.mkdir(parents=True,mode=0o700)
    for name,text in headers.items():(output/name).write_text(text)
    shutil.copyfile(weights,output/'weights.npz')
    if digest(output/'weights.npz')!=before or digest(weights)!=before:raise ValueError('numeric copy changed')
    (output/'decision-metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');return meta


class DscnnNativeAssets:
    def __init__(self,directory):
        directory=real_path(directory);meta=json.loads(real_path(directory/'decision-metadata.json').read_text())
        table=thresholds(meta['thresholds_q10']);layers,size=read_layers(directory/'weights.npz')
        if (meta.get('runtime')!=RUNTIME or meta.get('input_contract')!=CONTRACT or meta.get('frozen') is not False or
            meta.get('weights_sha256')!=digest(directory/'weights.npz') or meta.get('weights_bytes')!=size or meta.get('numeric_bytes')!=size+20):
            raise ValueError('numeric asset identity/contract changed')
        for name,text in render(layers,table).items():
            wanted=hashlib.sha256(text.encode()).hexdigest()
            if digest(real_path(directory/name))!=wanted or meta['header_sha256'].get(name)!=wanted:raise ValueError('header differs from regenerated numeric export')
        self.frontend=self.engine=None
        try:
            self.frontend=LogmelNativeFrontend();self.engine=DscnnNativeModel(layers)
        except BaseException:self.close();raise
        self.layers=layers;self.metadata=meta;self.weights_bytes=size;self.scratch_bytes=self.frontend.scratch_bytes
        self.table=np.array(table,np.int32);self.source_sha256={**self.frontend.source_sha256,**self.engine.source_sha256}
    def close(self):
        if self.frontend is not None:self.frontend.close()
        if self.engine is not None:self.engine.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def extract(self,pcm,*,banked=False):return self.frontend.extract(pcm,banked=banked)
    def predict(self,values):
        if not isinstance(values,np.ndarray) or values.shape!=(2113,):raise ValueError('one 2113-feature vector required')
        logits=self.engine.scores(values[None,:])[0];return self.engine.classify(logits,self.table),logits
