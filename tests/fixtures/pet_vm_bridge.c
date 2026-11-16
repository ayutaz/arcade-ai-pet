#include "guest_vm.h"
#include <stdlib.h>
static _Alignas(8) unsigned char pool[163840];
typedef struct {LabVm vm;int state,position,label,records,replies,stops;} Rig;
int pet_runtime_start(void){return lab_runtime_init(pool,sizeof(pool));}
void pet_runtime_stop(void){lab_runtime_destroy();}
static int audio(void *context,uint32_t op,uint32_t a,uint32_t b){
 Rig *r=context;(void)b;
 if(op==4)return r->state;
 if(op==7)return r->position;
 if(op==2){if(a!=2000 || r->state)return -1;r->state=2;r->position=0;r->records++;return 1;}
 if(op==9){if(r->state || r->position!=32000)return -1;r->position=0;return r->label;}
 if(op==10){if(a>4 || r->state)return -1;r->state=1;r->replies++;return 1;}
 if(op==6){r->state=r->position=0;r->stops++;return 1;}
 return -1;
}
void *pet_vm_open(uint8_t *blob,uint32_t size){Rig *r=calloc(1,sizeof(*r));if(!r)return NULL;LabVmAudio services={r,audio};
 if(!lab_vm_open_with_audio(&r->vm,blob,size,2,50000,7,&services)){free(r);return NULL;}return r;}
void pet_vm_free(void *context){Rig *r=context;lab_vm_close(&r->vm,1);free(r);}
int pet_vm_frame(void *context,uint32_t held,uint32_t dt){return lab_vm_frame(&((Rig*)context)->vm,held,dt);}
void pet_vm_voice(void *context,int label){Rig *r=context;r->state=0;r->position=32000;r->label=label;}
int pet_vm_stat(void *context,unsigned kind){Rig *r=context;return kind==0 ? lab_vm_debug(&r->vm,"debug_value"):kind==1 ? r->records:kind==2 ? r->replies:r->stops;}
