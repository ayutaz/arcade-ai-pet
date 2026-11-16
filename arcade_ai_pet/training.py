"""Small CPU-trained KWS MLP; evaluate held-out data only after model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .quantize import quantize_layer
from .features import split_for


def scores(x,layers):
    q=np.rint(np.clip(x,-16,16)*1024).astype(np.int64)
    for index,layer in enumerate(layers):
        q=(q@layer['weight'].astype(np.int64).T+layer['bias'])>>layer['shift']
        if index<len(layers)-1:q=np.clip(q,0,6144)
    return q


def metrics(y,prediction):
    confusion=np.zeros((7,7),dtype=np.int64)
    for actual,pred in zip(y,prediction,strict=True):confusion[actual,pred]+=1
    tp=np.diag(confusion);denominator=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*tp,denominator,out=np.zeros(7,dtype=float),where=denominator!=0)
    negative=confusion[5:].sum()
    return dict(macro_f1=float(f1.mean()),accuracy=float(tp.sum()/max(1,len(y))),
                false_activation_rate=float(confusion[5:,:5].sum()/max(1,negative)),confusion=confusion.tolist())


def decisions(logits,threshold):
    best=np.argmax(logits,axis=1)
    margins=np.sort(logits,axis=1)[:,-1]-np.sort(logits,axis=1)[:,-2]
    return np.where((best<5)&(margins<threshold),5,best)


def select_threshold(y,logits):
    candidates=[]
    for threshold in (0,128,256,512,768,1024,1536,2048,3072,4096):
        report=metrics(y,decisions(logits,threshold))
        candidates.append((report['false_activation_rate']>.05,-report['macro_f1'],threshold))
    return min(candidates)[2]


def load_dataset(dataset):
    dataset=Path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text())
    if type(manifest.get('variants')) is not int or not 1<=manifest['variants']<=10:
        raise ValueError('augmentation count')
    expected={split:[] for split in ('train','validation','test')};seen=set()
    for row in manifest['records']:
        source=row['source'];ident=row['id'];split=row['split'];label=row['label']
        if (type(source) is not str or hashlib.sha256(source.encode()).hexdigest()!=ident or ident in seen or
            split!=split_for(source) or type(label) is not int or not 0<=label<7):
            raise ValueError('source identity, leakage or split mismatch')
        seen.add(ident);expected[split].extend([label]*(manifest['variants'] if 'sha256' in row else 1))
    data={}
    for split,labels in expected.items():
        with np.load(dataset/(split+'.npz'),allow_pickle=False) as archive:
            x=archive['x'];y=archive['y']
        if (x.shape!=(len(labels),260) or not len(labels) or y.shape!=(len(labels),) or
            not np.issubdtype(y.dtype,np.integer) or not np.array_equal(y,labels) or
            not np.isfinite(x).all() or np.any(abs(x)>16)):
            raise ValueError('dataset feature/label contract')
        data[split]=dict(x=x,y=y)
    return data


def fit(x,y,vx,vy,*,seed=42,epochs=240,hidden=48):
    import torch
    for values,labels in ((x,y),(vx,vy)):
        if (not isinstance(values,np.ndarray) or values.ndim!=2 or values.shape[1]!=260 or
            not len(values) or labels.shape!=(len(values),) or not np.isfinite(values).all() or
            np.any(abs(values)>16) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>6))):
            raise ValueError('bounded features and labels required')
    if not 1<=epochs<=1000 or not 1<=hidden<=64:raise ValueError('training resource bounds')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=torch.nn.Sequential(torch.nn.Linear(260,hidden),torch.nn.ReLU6(),torch.nn.Linear(hidden,7))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    counts=np.bincount(y,minlength=7);weight=1/np.maximum(counts,1);weight/=weight.mean()
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32))
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.long)
    best=None;best_score=(-1,-1)
    for epoch in range(epochs):
        optimizer.zero_grad();loss=lossfn(model(tx),ty);loss.backward();optimizer.step()
        if epoch%5==0 or epoch==epochs-1:
            layers=[quantize_layer(model[index].weight.detach().numpy(),model[index].bias.detach().numpy()) for index in (0,2)]
            report=metrics(vy,scores(vx,layers).argmax(axis=1));key=(report['macro_f1'],report['accuracy'])
            if key>best_score:best_score=key;best=(layers,epoch+1,report)
    return dict(layers=best[0],selected_epoch=best[1],validation_accuracy=best[2]['accuracy'],
                validation=best[2],seed=seed,epochs=epochs)


def run(dataset,output,*,epochs=240,seed=42,hidden=48,validation_only=False):
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh training output required')
    data=load_dataset(dataset)
    result=fit(data['train']['x'],data['train']['y'],data['validation']['x'],data['validation']['y'],epochs=epochs,seed=seed,hidden=hidden)
    layers=result.pop('layers');logits=scores(data['validation']['x'],layers)
    threshold=select_threshold(data['validation']['y'],logits)
    result['threshold_q10']=threshold
    result['validation_selected']=metrics(data['validation']['y'],decisions(logits,threshold))
    # Test has no influence on checkpoint/threshold selection.
    result['weights_bytes']=sum(layer['weight'].nbytes+layer['bias'].nbytes+4 for layer in layers)
    result['test_evaluated']=not validation_only
    if not validation_only:
        result['test']=metrics(data['test']['y'],decisions(scores(data['test']['x'],layers),threshold))
        result['accepted_synthetic']=(result['test']['macro_f1']>=.90 and result['test']['false_activation_rate']<=.05 and result['weights_bytes']<=24576)
    result.update(real_voice_tested=False,device_tested=False,vast_usd=0,
                  dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
                  split_examples={split:len(data[split]['y']) for split in data})
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    (output/('validation.json' if validation_only else 'evaluation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--epochs',type=int,default=240)
    parser.add_argument('--hidden',type=int,default=48);parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.dataset,args.output,epochs=args.epochs,hidden=args.hidden,validation_only=args.validation_only)))
