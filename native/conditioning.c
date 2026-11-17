/* SPDX-License-Identifier: Apache-2.0
 * Experimental host candidate, not installed in the device firmware.
 * In-place PCM16, bounded sample count, fresh filter state per recording.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
int pet_condition(int16_t *pcm,size_t samples,int hz) {
    if(!pcm || samples!=32000 || (hz!=0 && hz!=20 && hz!=80 && hz!=160))return -1;
    if(hz) {
        float alpha=expf(-2.0f*3.14159265358979323846f*(float)hz/16000.0f);
        float previous=(float)pcm[0],value=0;
        for(size_t i=0;i<samples;i++) {
            float input=(float)pcm[i];value=alpha*(value+input-previous);previous=input;
            float clipped=fminf(32767,fmaxf(-32768,value));
            pcm[i]=(int16_t)roundf(clipped);
        }
    }
    int64_t sum=0;
    for(size_t i=0;i<samples;i++)sum+=pcm[i];
    int32_t mean=(int32_t)(sum/(int64_t)samples);
    for(size_t i=0;i<samples;i++) {
        int32_t value=(int32_t)pcm[i]-mean;
        pcm[i]=(int16_t)(value>32767 ? 32767:value<-32768 ? -32768:value);
    }
    return 1;
}
