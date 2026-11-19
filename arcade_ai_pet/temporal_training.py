"""Development-only temporal CNN training; no legacy freeze, C export or device I/O."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .channel_training import make_arrays
from .channel_robustness import PROFILES
from .domain_training import _inputs, validation_rank, select_threshold, shortlist
from .channel_freeze import _rows, _accepted
from .quantize import quantize_layer
from .temporal_model import sequence, scores, weights_bytes


def build_model():
    import torch
    class TemporalCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList((torch.nn.Conv1d(13,24,5,padding=2), torch.nn.Conv1d(24,32,3,padding=1), torch.nn.Conv1d(32,32,3,padding=1)))
            self.pool = torch.nn.AvgPool1d(4)
            self.fc = torch.nn.Linear(160,64)
            self.head = torch.nn.Linear(64,7)
        def forward(self, values):
            for layer in self.convs:
                values = torch.clamp(layer(values), 0, 6)
            values = self.pool(values).transpose(1,2).reshape(-1,160)
            return self.head(torch.clamp(self.fc(values), 0, 6))
    return TemporalCNN()


def quantize(model):
    layers = [quantize_layer(layer.weight.detach().cpu().numpy().reshape(layer.weight.shape[0],-1),
                             layer.bias.detach().cpu().numpy()) for layer in [*model.convs,model.fc,model.head]]
    weights_bytes(layers)
    return layers


def fit(x, y, vx, vy, groups, *, epochs=200, batch_size=256, seed=42, device='cpu'):
    import torch
    _inputs(vy, groups)
    for values, labels in ((x,y),(vx,vy)):
        sequence(values)
        if (not isinstance(labels,np.ndarray) or labels.shape != (len(values),) or
                not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>6))):
            raise ValueError('bounded training labels required')
    if (type(epochs) is not int or not 1<=epochs<=1000 or type(batch_size) is not int or
            not 1<=batch_size<=1024 or type(seed) is not int or not 0<=seed<2**31 or
            type(device) is not str or device not in ('cpu','mps')):
        raise ValueError('bounded CPU training configuration required')
    if device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('requested local MPS device unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    counts = np.bincount(y,minlength=7)
    weight = 1/np.maximum(counts,1)
    weight /= weight.mean()
    lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=device))
    tx = torch.tensor(sequence(x).transpose(0,2,1).copy(),dtype=torch.float32,device=device)
    ty = torch.tensor(y,dtype=torch.long,device=device)
    checkpoints = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y)).to(device)
        for start in range(0,len(y),batch_size):
            indexes = order[start:start+batch_size]
            optimizer.zero_grad()
            loss = lossfn(model(tx[indexes]),ty[indexes])
            loss.backward()
            optimizer.step()
        if epoch%5==0 or epoch==epochs-1:
            layers = quantize(model)
            logits = scores(vx,layers)
            checkpoints.append(dict(layers=layers,epoch=epoch+1,rank=validation_rank(vy,logits.argmax(1),groups)))
    candidates = shortlist(checkpoints)
    best = best_key = None
    for row in candidates:
        logits = scores(vx,row['layers'])
        threshold, reports, passed = select_threshold(vy,logits,groups)
        key = (passed,min(r['macro_f1'] for r in reports),sum(r['macro_f1'] for r in reports),-threshold,-row['epoch'])
        if best_key is None or key>best_key:
            best_key = key
            best = dict(layers=row['layers'],selected_epoch=row['epoch'],threshold_q10=threshold,
                        domain_profiles=reports,accepted_all_domains=passed,validation_accuracy=float(np.mean(logits.argmax(1)==vy)))
    best.update(seed=seed,epochs=epochs,batch_size=batch_size,training_device=device,checkpoint_candidates_evaluated=len(candidates),
                selection_metric='calibrated_worst_development_domain_profile')
    return best


def train_candidate(domains, mixed, output, *, epochs=200, batch_size=256, neural_weight=2, device='cpu'):
    output = Path(output).absolute()
    mixed = Path(mixed).absolute()
    if (output.resolve()!=output or output.exists() or type(domains) is not dict or set(domains)!={'legacy','neural'} or
            type(neural_weight) is not int or not 1<=neural_weight<=8 or type(device) is not str or device not in ('cpu','mps')):
        raise ValueError('fresh bounded development experiment required')
    domains = {name:Path(domains[name]) for name in ('legacy','neural')}
    records, seen, waves, variants = [], set(), set(), set()
    for path in domains.values():
        manifest = json.loads((path/'manifest.json').read_text())
        variants.add(manifest['variants'])
        if manifest.get('role','development')!='development':
            raise ValueError('final training data forbidden')
        for row in manifest['records']:
            if row['id'] in seen or 'sha256' in row and row['sha256'] in waves:
                raise ValueError('development domain overlap')
            seen.add(row['id'])
            if 'sha256' in row:
                waves.add(row['sha256'])
        records.extend(manifest['records'])
    combined = json.loads((mixed/'manifest.json').read_text())
    if len(variants)!=1 or combined.get('role')!='development' or combined['records']!=records or combined['variants'] not in variants:
        raise ValueError('mixed development identity differs from component domains')
    arrays = {name:make_arrays(path,'hp160',frontend='adaptive',noise_replicas=5,noise_mode='fresh',hum_replicas=3) for name,path in domains.items()}
    x = np.concatenate((arrays['legacy']['train']['x'], np.tile(arrays['neural']['train']['x'],(neural_weight,1))))
    y = np.concatenate((arrays['legacy']['train']['y'], np.tile(arrays['neural']['train']['y'],neural_weight)))
    vx = np.concatenate([a['validation']['x'] for a in arrays.values()])
    vy = np.concatenate([a['validation']['y'] for a in arrays.values()])
    profiles = [p for a in arrays.values() for p in a['validation']['profiles']]
    groups = np.array([domain*8+PROFILES.index(p) for domain,a in enumerate(arrays.values()) for p in a['validation']['profiles']])
    result = fit(x,y,vx,vy,groups,epochs=epochs,batch_size=batch_size,device=device)
    layers = result.pop('layers')
    logits = scores(vx,layers)
    threshold, rows, accepted = select_threshold(vy,logits,groups)
    pooled = _rows(vy,logits,profiles,'hp160',threshold)
    result.update(classifier='temporal-cnn-v1',training_device=device,candidate='hp160',frontend_version='adaptive',threshold_q10=threshold,
                  validation=pooled,domain_profiles=rows,accepted_validation=accepted and _accepted(pooled),
                  training_domain_weights=dict(legacy=1,neural=neural_weight),noise_replicas=5,hum_replicas=3,noise_mode='fresh',
                  dataset_manifest_sha256=hashlib.sha256((mixed/'manifest.json').read_bytes()).hexdigest(),
                  trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  feature_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('channel_frontend.py','channel_training.py','temporal_model.py')},
                  weights_bytes=weights_bytes(layers),split_examples=dict(train=len(y),validation=len(vy)),
                  final_evaluated=False,real_voice_tested=False,device_tested=False,vast_usd=0,
                  legacy_export_supported=False)
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    result['weights_sha256'] = hashlib.sha256((output/'weights.npz').read_bytes()).hexdigest()
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
