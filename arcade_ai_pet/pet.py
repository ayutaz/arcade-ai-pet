"""Companion state independent of audio I/O, GUI and the learned policy trainer."""
from pathlib import Path
from .behavior import World,MOVES,observation
from .behavior_native import load_behavior
from .quantize import predict

ROOT=Path(__file__).resolve().parents[1]


class Pet:
    def __init__(self,seed=17):
        self.seed=seed;self.world=World(seed);self.layers=load_behavior(ROOT/'models/behavior-v1/weights.npz')
        self.mode='idle';self.reply=-1;self.reply_serial=0;self.closed=False;self.accumulator=0
        self.player=next((i%8,i//8) for i in range(47,-1,-1) if (i%8,i//8) not in self.world.walls)
        self.player_score=0;self.affection=0
    def command(self,action):
        if type(action) is not int:raise ValueError('integer action')
        if self.closed or not 0<=action<5:return
        self.mode=('play','come','wait','sleep','happy')[action]
        self.reply=action;self.reply_serial+=1
        if action==4:self.affection=min(99,self.affection+1)
        self.accumulator=0
    def button(self,pressed):
        if type(pressed) is not int or not 0<=pressed<=255:raise ValueError('button mask')
        if self.closed:return
        if pressed&64:self.closed=True;return
        if pressed&32:self.command(0);return
        if self.mode=='play':
            # Same priority as the public game SDK reference.
            action=next((a for bit,a in ((1,1),(2,0),(4,3),(8,2)) if pressed&bit),None)
            if action is not None:
                dx,dy=MOVES[action];x,y=self.player[0]+dx,self.player[1]+dy
                if 0<=x<8 and 0<=y<6 and (x,y) not in self.world.walls:self.player=x,y
                if self.player==self.world.goal:
                    self.player_score+=1
                    for _ in range(256):
                        self.world.new_goal()
                        if self.world.goal!=self.player:break
                    else:raise RuntimeError('new shared target')
        else:
            for bit,action in ((8,1),(2,2),(4,3),(1,4)):
                if pressed&bit:self.command(action);break
    def tick(self,dt):
        if type(dt) is not int or not 0<=dt<=60000:raise ValueError('bounded time')
        if self.closed or self.mode!='play':return
        # At most one model step per frame; discard catch-up after a long pause.
        self.accumulator+=min(dt,200)
        if self.accumulator>=200:
            self.accumulator-=200;self.world.step(predict(observation(self.world),self.layers))
    def snapshot(self):
        return self.mode,self.reply,self.reply_serial,self.closed,self.player,self.player_score,self.affection,self.world.snapshot(),self.accumulator
