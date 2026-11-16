"""Small deterministic collection world and independent policy baselines."""
from collections import deque
import numpy as np

MOVES=((-1,0),(1,0),(0,-1),(0,1))


class World:
    width=8;height=6
    def __init__(self,seed):
        if type(seed) is not int or not 0<=seed<=0xffffffff:raise ValueError('u32 seed')
        self.rng=seed;self.x=0;self.y=0;self.score=0;self.steps=0;self.walls=set()
        while len(self.walls)<4:
            cell=self.cell()
            if cell==(0,0) or cell in self.walls:continue
            candidate=self.walls|{cell}
            seen={(0,0)};queue=deque(seen)
            while queue:
                x,y=queue.popleft()
                for dx,dy in MOVES:
                    n=x+dx,y+dy
                    if 0<=n[0]<8 and 0<=n[1]<6 and n not in candidate and n not in seen:
                        seen.add(n);queue.append(n)
            if len(seen)==48-len(candidate):self.walls=candidate
        self.new_goal()
    def number(self):
        self.rng=(1664525*self.rng+1013904223)&0xffffffff
        return self.rng>>8
    def cell(self):
        n=self.number()%48;return n%8,n//8
    def new_goal(self):
        for _ in range(256):
            cell=self.cell()
            if cell not in self.walls and cell!=(self.x,self.y):self.goal=cell;return
        raise RuntimeError('goal sampling exhausted')
    def blocked(self,action):
        dx,dy=MOVES[action];x,y=self.x+dx,self.y+dy
        return not (0<=x<8 and 0<=y<6) or (x,y) in self.walls
    def step(self,action):
        if type(action) is not int or not 0<=action<4:raise ValueError('four actions')
        self.steps+=1
        if not self.blocked(action):
            dx,dy=MOVES[action];self.x+=dx;self.y+=dy
        if (self.x,self.y)==self.goal:self.score+=1;self.new_goal()
    def snapshot(self):return self.x,self.y,self.goal,tuple(sorted(self.walls)),self.score,self.steps,self.rng


def observation(world):
    # Integer divisions are shared with the target; convert Q10 exactly to float.
    def trunc(n,d):return (1 if n>=0 else -1)*(abs(n)//d)
    return np.array([trunc((world.goal[0]-world.x)*1024,7),trunc((world.goal[1]-world.y)*1024,5),
                     *(1024*world.blocked(i) for i in range(4))],dtype=np.float32)/1024


def teacher(world):
    queue=deque([(world.x,world.y,-1)]);seen={(world.x,world.y)}
    while queue:
        x,y,first=queue.popleft()
        if (x,y)==world.goal:return first
        for action,(dx,dy) in enumerate(MOVES):
            p=x+dx,y+dy
            if 0<=p[0]<8 and 0<=p[1]<6 and p not in world.walls and p not in seen:
                seen.add(p);queue.append((*p,action if first<0 else first))
    raise ValueError('unreachable goal')


def greedy_action(world):
    return min(range(4),key=lambda i:(world.blocked(i),abs(world.x+MOVES[i][0]-world.goal[0])+abs(world.y+MOVES[i][1]-world.goal[1]),i))


def random_action(world):
    # Separate action RNG, never mutates the world/goal stream.
    n=(world.rng^((world.steps+1)*2654435761))&0xffffffff
    n^=n>>16;n=(n*2246822519)&0xffffffff;n^=n>>13
    return n%4


def benchmark(policy,*,seeds,steps=120):
    seeds=list(seeds)
    if not seeds or len(seeds)>4096 or len(set(seeds))!=len(seeds) or type(steps) is not int or not 1<=steps<=1000:
        raise ValueError('unique seeds and bounded rollout')
    scores=[]
    for seed in seeds:
        world=World(seed)
        for _ in range(steps):world.step(int(policy(world)))
        scores.append(world.score)
    return dict(episodes=len(scores),steps=steps,mean_score=float(np.mean(scores)),scores=scores,seeds=seeds)
