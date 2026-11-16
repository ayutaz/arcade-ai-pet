"""Fixed-shape numeric export for the independently trained movement policy."""
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from .native import _array


def load_behavior(path):
    path=Path(path)
    if path.stat().st_size>65536:raise ValueError('bounded policy file')
    expected={f'{k}{i}.npy' for i in range(2) for k in ('weight','bias','shift')}
    try:
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist())!=expected or len(archive.infolist())!=6:raise ValueError('policy members')
            for member in archive.infolist():
                if member.file_size>8192:raise ValueError('bounded member')
                with archive.open(member) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                    elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                    else:raise ValueError('array format')
                    index=int(member.filename[-5]);field=member.filename[:-5]
                    wanted={'weight':((32,6),(4,32)),'bias':((32,),(4,)),'shift':((),())}[field][index]
                    if shape!=wanted or order or dtype.hasobject or member.file_size-stream.tell()!=int(np.prod(shape,dtype=np.int64))*dtype.itemsize:
                        raise ValueError('policy shape/dtype/extent')
        with np.load(path,allow_pickle=False) as archive:
            layers=[]
            for i in range(2):
                layer={k:archive[f'{k}{i}'] for k in ('weight','bias','shift')}
                if layer['weight'].dtype!=np.int8 or layer['bias'].dtype!=np.int32 or not np.issubdtype(layer['shift'].dtype,np.integer):
                    raise ValueError('integer policy')
                layer['shift']=int(layer['shift'])
                if not 0<=layer['shift']<=16:raise ValueError('policy shift')
                layers.append(layer)
            return layers
    except (OSError,KeyError,zipfile.BadZipFile,EOFError) as error:raise ValueError('policy archive') from error


def export_behavior(source,output):
    output=Path(output)
    if output.exists():raise ValueError('fresh policy export')
    layers=load_behavior(source)
    text='/* SPDX-License-Identifier: Apache-2.0; learned BFS-imitation policy. */\n'
    for i,layer in enumerate(layers):
        text+=f'enum {{ BEHAVIOR_SHIFT{i}={layer["shift"]} }};\n'
        text+=_array(f'behavior_weight{i}',layer['weight'],'int8_t')+_array(f'behavior_bias{i}',layer['bias'],'int32_t')
    output.mkdir(parents=True);(output/'behavior_model.h').write_text(text)
    return dict(weights_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),weights_bytes=472)
