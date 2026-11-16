"""A fixed 30-trial protocol and anonymous result summary; no device/audio access."""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys

COMMANDS=('あそぼ','おいで','まって','おやすみ','ありがとう')
NEGATIVES=('こんにちは','さようなら','いまなんじ','おはよう','こんばんは','てんき','でんわ','ほん','みず','ねこ')


def trial_plan():
    prompts=[(label,word) for _ in range(3) for label,word in enumerate(COMMANDS)]
    prompts.extend((5,word) for word in NEGATIVES)
    prompts.extend((6,'2秒間、話さずに待つ') for _ in range(5))
    return [dict(trial=index,expected=label,prompt=word) for index,(label,word) in enumerate(prompts,1)]


def summarize(data):
    if (type(data) is not dict or set(data)!={'schema','scope','records'} or
        type(data['schema']) is not int or data['schema']!=1 or
        data['scope'] not in ('host_fixture','device_synthetic','device_human') or
        type(data['records']) is not list or len(data['records'])!=30):
        raise ValueError('complete fixed trial results required')
    plan={row['trial']:row for row in trial_plan()};seen=set();matrix=[[0]*7 for _ in range(7)]
    correct=false_activations=errors=0;latencies=[]
    for row in data['records']:
        if type(row) is not dict or set(row)!={'trial','observed','latency_ms'}:
            raise ValueError('anonymous numeric trial fields required')
        trial,observed,delay=row['trial'],row['observed'],row['latency_ms']
        if (type(trial) is not int or trial not in plan or trial in seen or
            type(observed) is not int or not -1<=observed<=6 or
            type(delay) not in (int,float) or not 0<=delay<=60000 or not math.isfinite(delay)):
            raise ValueError('trial values outside the protocol')
        seen.add(trial);expected=plan[trial]['expected'];latencies.append(delay)
        if observed==-1:errors+=1
        else:matrix[expected][observed]+=1
        if expected<5 and observed==expected:correct+=1
        if expected>=5 and 0<=observed<5:false_activations+=1
    return dict(schema=1,reported_scope=data['scope'],complete=True,command_correct=correct,
                command_total=15,negative_total=15,false_activations=false_activations,errors=errors,
                thresholds=dict(command_correct_min=12,false_activations_max=1,errors_max=0),
                thresholds_met=correct>=12 and false_activations<=1 and errors==0,
                hardware_acceptance_proven=False,confusion=matrix,
                latency_ms=dict(min=min(latencies),median=statistics.median(latencies),max=max(latencies)))


def _read(path):
    fd=os.open(Path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 1<=info.st_size<=16384:
            raise ValueError('bounded regular result file required')
        raw=stream.read(16385)
    if len(raw)!=info.st_size:raise ValueError('result file changed')
    return json.loads(raw)


def main(argv=None,*,out=None):
    out=out or sys.stdout;parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True);commands.add_parser('plan')
    command=commands.add_parser('summarize');command.add_argument('results')
    args=parser.parse_args(argv)
    try:
        result=dict(schema=1,trials=trial_plan()) if args.action=='plan' else summarize(_read(args.results))
        print(json.dumps(result,ensure_ascii=False),file=out);return 0
    except (OSError,ValueError,TypeError,OverflowError):
        print('Trial results are invalid or unavailable.',file=out);return 2


if __name__=='__main__':raise SystemExit(main())
