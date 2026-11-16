#include <stdint.h>
#include "app.c"
static uint32_t previous;
static int state,position,recognized,records,replies,stops,bad_draw;
void rect(int32_t x,int32_t y,int32_t w,int32_t h,int32_t color){if(x<0 || y<0 || w<1 || h<1 || x+w>320 || y+h>240 || color<0 || color>65535)bad_draw=1;}
int32_t audio_state(void){return state;}
int32_t audio_position(void){return position;}
int32_t audio_record(int32_t ms){if(state || ms!=2000)return -1;state=2;position=0;records++;return 1;}
int32_t audio_classify(void){if(state || position!=32000)return -1;position=0;return recognized;}
int32_t audio_reply(int32_t id){if(state || id<0 || id>4)return -1;state=1;replies++;return 1;}
int32_t audio_stop(void){state=0;position=0;stops++;return 1;}
void test_init(void){previous=0;state=position=records=replies=stops=bad_draw=0;recognized=6;init();}
int test_tick(uint32_t held,uint32_t dt){uint32_t pressed=held&~previous,released=previous&~held;previous=held;return tick(held,pressed,released,dt);}
void test_voice(int command){state=0;position=32000;recognized=command;}
void test_idle(void){state=0;}
int test_stat(unsigned n){return n==0 ? records:n==1 ? replies:n==2 ? stops:bad_draw;}
void test_seed(unsigned seed){reset_world(seed);}
void test_action(unsigned id){apply_command(id);}
