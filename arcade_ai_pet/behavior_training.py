"""CPU imitation learning from BFS; select only on held-out validation seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .behavior import World,observation,teacher,benchmark,random_action,greedy_action
from .quantize import quantize_layer,predict


def dataset(seeds,*,samples=64):
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or not 1<=samples<=256:raise ValueError('unique data seeds and sample bound')
    x=[];y=[]
    for seed in seeds:
        world=World(seed)
        for _ in range(samples):
            # Independent free positions broaden states beyond expert rollouts.
            for _ in range(256):
                position=world.cell()
                if position not in world.walls and position!=world.goal:world.x,world.y=position;break
            else:raise RuntimeError('state sampling')
            x.append(observation(world));y.append(teacher(world))
            world.step(y[-1])
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.int64)


def action_for(layers):return lambda world:predict(observation(world),layers)


def fit_behavior(x,y,vx,vy,*,epochs=300,seed=73):
    import torch
    for values,labels in ((x,y),(vx,vy)):
        if (values.ndim!=2 or values.shape[1]!=6 or not len(values) or labels.shape!=(len(values),) or
            not np.isfinite(values).all() or np.any(abs(values)>1) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>3))):
            raise ValueError('finite six-feature four-action data')
    if type(epochs) is not int or not 1<=epochs<=1000:raise ValueError('epoch bound')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=torch.nn.Sequential(torch.nn.Linear(6,32),torch.nn.ReLU6(),torch.nn.Linear(32,4))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0001)
    tx=torch.tensor(x);ty=torch.tensor(y);best=None;accuracy=-1
    for epoch in range(epochs):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(tx),ty);loss.backward();optimizer.step()
        if epoch%5==0 or epoch==epochs-1:
            layers=[quantize_layer(model[i].weight.detach().numpy(),model[i].bias.detach().numpy()) for i in (0,2)]
            q=np.rint(vx*1024).astype(np.int64)
            for i,layer in enumerate(layers):
                q=(q@layer['weight'].astype(np.int64).T+layer['bias'])>>layer['shift']
                if i==0:q=np.clip(q,0,6144)
            score=float(np.mean(q.argmax(1)==vy))
            if score>accuracy:accuracy=score;best=(layers,epoch+1)
    return dict(layers=best[0],selected_epoch=best[1],validation_accuracy=accuracy,seed=seed,epochs=epochs,
                weights_bytes=sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in best[0]))


def run(output,*,epochs=300):
    output=Path(output)
    if output.exists():raise ValueError('fresh output required')
    train=list(range(256));validation=list(range(10000,10032));test=list(range(20000,20064))
    x,y=dataset(train);vx,vy=dataset(validation)
    result=fit_behavior(x,y,vx,vy,epochs=epochs);layers=result.pop('layers')
    output.mkdir(parents=True)
    np.savez(output/'weights.npz',**{f'{k}{i}':l[k] for i,l in enumerate(layers) for k in ('weight','bias','shift')})
    # Model is fixed on disk before test seed rollouts; no test-driven selection.
    result.update(weights_sha256=hashlib.sha256((output/'weights.npz').read_bytes()).hexdigest(),
                  learning='supervised imitation of BFS, not reinforcement learning',features='relative-goal-q10+blocked4-v1',
                  train_seeds=train,validation_seeds=validation,test_seeds=test,host_only=True,device_tested=False,vast_usd=0)
    result['test']={name:benchmark(policy,seeds=test) for name,policy in (
        ('learned',action_for(layers)),('random',random_action),('greedy',greedy_action),('bfs_upper_reference',teacher))}
    result['accepted']=result['test']['learned']['mean_score']>result['test']['random']['mean_score']
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args();result=run(args.output,epochs=args.epochs)
    print(json.dumps({k:v for k,v in result.items() if k not in ('train_seeds','validation_seeds','test_seeds','test')}))
    print(json.dumps({name:row['mean_score'] for name,row in result['test'].items()}))
