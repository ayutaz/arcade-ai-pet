"""Freeze validation-selected weights before generating independent evaluation sources."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile
import numpy as np
from .training import load_dataset, scores, decisions, metrics


def read_layers(path):
    if Path(path).stat().st_size>262144:raise ValueError('bounded model archive required')
    try:
        with zipfile.ZipFile(path) as zipped:
            names={f'{name}{i}.npy' for i in (0,1) for name in ('weight','bias','shift')}
            if len(zipped.infolist())!=6 or set(zipped.namelist())!=names:raise ValueError('integer model members')
            for info in zipped.infolist():
                if info.file_size>65536:raise ValueError('model array budget')
                with zipped.open(info) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=2048)
                    elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=2048)
                    else:raise ValueError('model array format')
                    if info.filename.startswith('weight'):
                        valid=dtype==np.int8 and len(shape)==2 and 1<=shape[0]<=64 and 1<=shape[1]<=260
                    elif info.filename.startswith('bias'):
                        valid=dtype==np.int32 and len(shape)==1 and 1<=shape[0]<=64
                    else:valid=np.issubdtype(dtype,np.integer) and dtype.itemsize<=8 and shape==()
                    if not valid or stream.tell()+math.prod(shape)*dtype.itemsize!=info.file_size:
                        raise ValueError('model array shape/payload budget')
    except zipfile.BadZipFile:raise ValueError('model archive required') from None
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={f'{name}{i}' for i in (0,1) for name in ('weight','bias','shift')}:
            raise ValueError('two-layer integer model required')
        layers=[]
        for index in (0,1):
            weight=archive[f'weight{index}'];bias=archive[f'bias{index}'];shift=archive[f'shift{index}']
            if (weight.dtype!=np.int8 or weight.ndim!=2 or bias.dtype!=np.int32 or
                bias.shape!=(weight.shape[0],) or shift.shape!=() or not np.issubdtype(shift.dtype,np.integer) or
                not 0<=int(shift)<=16):raise ValueError('integer model representation')
            layers.append(dict(weight=weight,bias=bias,shift=int(shift)))
    hidden=layers[0]['weight'].shape[0]
    if not 1<=hidden<=64 or layers[0]['weight'].shape!=(hidden,260) or layers[1]['weight'].shape!=(7,hidden):
        raise ValueError('bounded model architecture')
    size=sum(layer['weight'].nbytes+layer['bias'].nbytes+4 for layer in layers)
    if size>24576:raise ValueError('24KiB weight budget exceeded')
    return layers,size


def freeze_model(checkpoint,development,output):
    checkpoint=Path(checkpoint).resolve();development=Path(development).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh frozen model directory required')
    metadata=json.loads((checkpoint/'validation.json').read_text());validation=metadata['validation_selected']
    f1=validation['macro_f1'];false=validation['false_activation_rate'];threshold=metadata['threshold_q10']
    if (type(f1) not in (int,float) or not math.isfinite(f1) or not .9<=f1<=1 or
        type(false) not in (int,float) or not math.isfinite(false) or not 0<=false<=.05 or
        metadata.get('test_evaluated') is not False or type(threshold) is not int or not 0<=threshold<=4096 or
        type(metadata.get('selected_epoch')) is not int or not 1<=metadata['selected_epoch']<=1000):
        raise ValueError('validation-selected candidate required')
    development_data=load_dataset(development)
    manifest=json.loads((development/'manifest.json').read_text())
    if manifest.get('feature_version')!='aligned-mfcc-v2':raise ValueError('feature version mismatch')
    layers,size=read_layers(checkpoint/'weights.npz')
    independent=metrics(development_data['validation']['y'],decisions(scores(development_data['validation']['x'],layers),threshold))
    if independent['macro_f1']<.9 or independent['false_activation_rate']>.05:
        raise ValueError('independent validation does not qualify the candidate')
    result=dict(schema=1,architecture=[260,layers[0]['weight'].shape[0],7],weights_bytes=size,
                weights_sha256=hashlib.sha256((checkpoint/'weights.npz').read_bytes()).hexdigest(),
                development_manifest_sha256=hashlib.sha256((development/'manifest.json').read_bytes()).hexdigest(),
                validation_sha256=hashlib.sha256((checkpoint/'validation.json').read_bytes()).hexdigest(),
                source_ids=sorted(row['id'] for row in manifest['records']),
                voiced_file_hashes=sorted({row['sha256'] for row in manifest['records'] if row['label']<6 and 'sha256' in row}),
                threshold_q10=threshold,selected_epoch=metadata['selected_epoch'],feature_version='aligned-mfcc-v2',
                license='CC-BY-3.0; see docs/data-license.md',real_voice_tested=False,device_tested=False)
    output.mkdir(parents=True,mode=0o700);shutil.copyfile(checkpoint/'weights.npz',output/'weights.npz')
    (output/'frozen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def evaluate_frozen(frozen,dataset,output):
    frozen=Path(frozen).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh evaluation output required')
    model=json.loads((frozen/'frozen.json').read_text())
    actual_hash=hashlib.sha256((frozen/'weights.npz').read_bytes()).hexdigest()
    if actual_hash!=model['weights_sha256']:raise ValueError('frozen weights changed')
    layers,size=read_layers(frozen/'weights.npz')
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest.get('feature_version')!=model['feature_version']:raise ValueError('feature version mismatch')
    overlap=set(model['source_ids'])&{row['id'] for row in manifest['records']}
    if overlap:raise ValueError('evaluation reuses development source utterances')
    wave_overlap=set(model['voiced_file_hashes'])&{row['sha256'] for row in manifest['records'] if row['label']<6 and 'sha256' in row}
    if wave_overlap:raise ValueError('evaluation reuses identical voiced source files')
    data=load_dataset(dataset)
    x=np.concatenate([data[split]['x'] for split in ('train','validation','test')])
    y=np.concatenate([data[split]['y'] for split in ('train','validation','test')])
    if set(y.tolist())!=set(range(7)):raise ValueError('all seven evaluation classes required')
    result=metrics(y,decisions(scores(x,layers),model['threshold_q10']))
    result.update(weights_sha256=actual_hash,weights_bytes=size,source_overlap=0,examples=len(y),
                  original_sources=len(manifest['records']),evaluation_scope='all newly generated sources; no training',
                  dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
                  threshold_q10=model['threshold_q10'],real_voice_tested=False,device_tested=False,vast_usd=0)
    result['accepted_synthetic']=result['macro_f1']>=.90 and result['false_activation_rate']<=.05
    output.mkdir(parents=True)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
