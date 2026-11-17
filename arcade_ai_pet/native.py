"""Deterministic C tables/model export; never a firmware deployment command."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import _WINDOW,_MEL,_DCT
from .frozen import read_layers


def _array(name,values,ctype):
    values=np.asarray(values)
    def scalar(value):return float(np.float32(value)).hex()+'f' if ctype=='float' else str(int(value))
    def body(array):
        return '{'+','.join(body(row) for row in array)+'}' if array.ndim>1 else '{'+','.join(scalar(v) for v in array)+'}'
    dimensions=''.join(f'[{size}]' for size in values.shape)
    return f'static const {ctype} {name}{dimensions} = '+body(values)+';\n'


def _render_tables(layers,threshold):
    indices=np.full((257,2),-1,dtype=np.int8);weights=np.zeros((257,2),dtype=np.float32)
    for index in range(257):
        active=np.flatnonzero(_MEL[:,index]>0)
        if len(active)>2:raise ValueError('sparse mel filter contract')
        indices[index,:len(active)]=active;weights[index,:len(active)]=_MEL[active,index]
    tables='/* SPDX-License-Identifier: Apache-2.0; generated mathematical tables. */\n'
    for name,array,ctype in (('pet_window',_WINDOW,'float'),('pet_mel_index',indices,'int8_t'),
            ('pet_mel_weight',weights,'float'),('pet_dct',_DCT,'float'),
            ('pet_fft_cos',np.cos(2*np.pi*np.arange(256)/512),'float'),
            ('pet_fft_sin',np.sin(2*np.pi*np.arange(256)/512),'float')):
        tables+=_array(name,array,ctype)
    model='''/* SPDX-License-Identifier: CC-BY-3.0
 * Integer KWS model trained from synthetic HTS Voice Mei / NIT ATR503 M001 speech.
 * Nagoya Institute of Technology / Tokyo Institute of Technology voice attribution,
 * generation/training/quantization details: docs/data-license.md and model card.
 * No human or device recording was used. This is not a real-voice accuracy claim.
 */
'''
    model+=f'enum {{ PET_HIDDEN={layers[0]["weight"].shape[0]}, PET_THRESHOLD={threshold} }};\n'
    for index,layer in enumerate(layers):
        model+=f'enum {{ PET_SHIFT{index}={layer["shift"]} }};\n'
        model+=_array(f'pet_weight{index}',layer['weight'],'int8_t')
        model+=_array(f'pet_bias{index}',layer['bias'],'int32_t')
    return tables,model


def export_assets(frozen,output):
    frozen=Path(frozen).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh native asset output required')
    metadata=json.loads((frozen/'frozen.json').read_text())
    digest=hashlib.sha256((frozen/'weights.npz').read_bytes()).hexdigest()
    if digest!=metadata['weights_sha256'] or metadata['feature_version']!='aligned-mfcc-v2':
        raise ValueError('frozen model identity/version mismatch')
    if type(metadata['threshold_q10']) is not int or not 0<=metadata['threshold_q10']<=4096:
        raise ValueError('bounded integer threshold required')
    layers,size=read_layers(frozen/'weights.npz')
    tables,model=_render_tables(layers,metadata["threshold_q10"])
    output.mkdir(parents=True)
    for name,text in (('feature_tables.h',tables),('pet_model.h',model)):(output/name).write_text(text)
    result=dict(feature_version='aligned-mfcc-v2',weights_sha256=digest,weights_bytes=size,threshold_q10=metadata['threshold_q10'],
        asset_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('feature_tables.h','pet_model.h')},
        device_tested=False)
    (output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n');return result
