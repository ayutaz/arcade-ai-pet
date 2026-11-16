/* SPDX-License-Identifier: Apache-2.0 */
#include "lab_os.h"
#include "lab_digits.h"
#include "behavior.c"
static uint32_t rng,steps,score,player_score,mode,affection,accumulator;
static int x,y,gx,gy,px,py,closed,recording,pending_reply;
static uint8_t walls[48];
static const int dx[4]={-1,1,0,0},dy[4]={0,0,-1,1};
static unsigned cell(void){rng=1664525u*rng+1013904223u;return (rng>>8)%48;}
static int free_cell(int a,int b){return a>=0 && a<8 && b>=0 && b<6 && !walls[b*8+a];}
static int connected(unsigned count){
    (void)count;
    const uint64_t board=UINT64_C(0xffffffffffff),left=UINT64_C(0x010101010101),right=UINT64_C(0x808080808080);
    uint64_t allowed=board,seen=1;
    for(unsigned i=0;i<48;i++)if(walls[i])allowed&=~(UINT64_C(1)<<i);
    for(unsigned i=0;i<48;i++){
        uint64_t next=(seen|(seen<<8)|(seen>>8)|((seen&~left)>>1)|((seen&~right)<<1))&allowed;
        if(next==seen)return seen==allowed;
        seen=next;
    }
    return seen==allowed;
}
static void new_goal(void){
    for(unsigned i=0;i<256;i++){unsigned at=cell();if(!walls[at] && ((int)(at%8)!=x || (int)(at/8)!=y)){gx=(int)(at%8);gy=(int)(at/8);return;}}
    closed=1;
}
static void reset_world(uint32_t seed){
    rng=seed;x=y=0;score=steps=player_score=accumulator=0;
    for(unsigned i=0;i<48;i++)walls[i]=0;
    unsigned count=0;
    for(unsigned attempts=0;attempts<1024 && count<4;attempts++){
        unsigned at=cell();if(!at || walls[at])continue;
        walls[at]=1;if(connected(count+1))count++;else walls[at]=0;
    }
    if(count!=4){closed=1;return;}
    new_goal();for(int at=47;at>=0;at--)if(!walls[at]){px=at%8;py=at/8;break;}
}
static void apply_command(unsigned action){
    if(closed || action>=5)return;
    mode=action+1;pending_reply=(int)action;accumulator=0;
    if(action==4 && affection<99)affection++;
}
static void companion_step(void){
    int32_t features[6]={(gx-x)*1024/7,(gy-y)*1024/5,0,0,0,0},scores[4];
    for(unsigned i=0;i<4;i++)features[2+i]=free_cell(x+dx[i],y+dy[i]) ? 0:1024;
    int action=pet_behavior(features,6,scores);if(action<0){closed=1;return;}
    steps++;
    if(free_cell(x+dx[action],y+dy[action])){x+=dx[action];y+=dy[action];}
    if(x==gx && y==gy){score++;new_goal();}
}
static void player_move(uint32_t pressed){
    int action=pressed&RIGHT ? 1:pressed&LEFT ? 0:pressed&DOWN ? 3:pressed&UP ? 2:-1;
    if(action<0)return;
    int a=px+dx[action],b=py+dy[action];if(free_cell(a,b)){px=a;py=b;}
    if(px==gx && py==gy){player_score++;for(unsigned i=0;i<256;i++){new_goal();if(px!=gx || py!=gy)break;}}
}
static void paint(void){
    const int colors[6]={31727,2016,2047,65504,14823,63519};
    rect(0,0,320,240,2113);rect(12,8,296,8,colors[mode]);
    if(recording)rect(288,24,12,12,63488);
    if(mode==1){
        for(unsigned i=0;i<48;i++)if(walls[i])rect(21+(int)(i%8)*20,41+(int)(i/8)*20,18,18,27469);
        rect(26+gx*20,46+gy*20,8,8,65504);
        rect(22+x*20,42+y*20,16,16,2016);rect(25+x*20,45+y*20,3,3,0);rect(32+x*20,45+y*20,3,3,0);
        rect(23+px*20,43+py*20,14,14,31);
        lab_number((int)(score%100),20,168);lab_number((int)(player_score%100),116,168);
    } else {
        int side=mode==2 ? 80:64;int left=160-side/2,top=88-side/2;
        rect(left,top,side,side,colors[mode]);
        rect(left-8,top-12,16,16,colors[mode]);rect(left+side-8,top-12,16,16,colors[mode]);
        int eye=mode==4 ? 3:8;
        rect(left+12,top+20,8,eye,0);rect(left+side-20,top+20,8,eye,0);
        rect(left+side/2-8,top+side-20,16,mode==5 ? 8:3,0);
        lab_number((int)affection,144,160);
    }
    /* Five small action indicators; voice replies explain the selected action. */
    for(unsigned i=0;i<5;i++)rect(240,48+(int)i*24,20,12,mode==i+1 ? colors[mode]:14823);
}
uint32_t sdk_abi(void){return LAB_ABI;}
void init(void){closed=recording=0;pending_reply=-1;mode=affection=0;reset_world(17);}
void shutdown(uint32_t reason){(void)reason;if(!closed)audio_stop();closed=1;}
int32_t debug_value(void){return (int32_t)((mode<<24)|((player_score&4095)<<12)|(score&4095));}
int32_t debug_x(void){return x+px*256;}
int32_t debug_y(void){return y+py*256;}
int32_t debug_elapsed(void){return (int32_t)steps;}
int32_t tick(uint32_t held,uint32_t pressed,uint32_t released,uint32_t dt){
    (void)held;(void)released;
    if(closed)return 1;
    if(pressed&BACK){audio_stop();closed=1;return 1;}
    int state=audio_state();
    if(recording){
        if(state<0)recording=0;
        else if(state==0){
            recording=0;
            if(audio_position()==32000){int action=audio_classify();if(action>=0 && action<5)apply_command((unsigned)action);}
        }
    } else if((pressed&START) && state==0){
        if(audio_record(2000)==1){recording=1;pending_reply=-1;}
    }
    if(pressed&RESET)apply_command(0);
    else if(mode==1)player_move(pressed);
    else if(pressed&UP)apply_command(1);
    else if(pressed&LEFT)apply_command(2);
    else if(pressed&DOWN)apply_command(3);
    else if(pressed&RIGHT)apply_command(4);
    if(!recording && pending_reply>=0 && state==0){
        int result=audio_reply(pending_reply);if(result!=0)pending_reply=-1;
    }
    if(mode==1){accumulator+=dt>200 ? 200:dt;if(accumulator>=200){accumulator-=200;companion_step();}}
    paint();return closed ? 1:0;
}
