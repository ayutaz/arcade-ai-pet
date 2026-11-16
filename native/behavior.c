/* SPDX-License-Identifier: Apache-2.0 */
#include <stdint.h>
#include <stddef.h>
#include "behavior_model.h"
static int64_t down(int64_t value,unsigned shift) {
    int64_t divisor=INT64_C(1)<<shift;
    return value>=0 ? value/divisor : -((-value+divisor-1)/divisor);
}
int pet_behavior(const int32_t *q10,size_t count,int32_t *scores) {
    if(!q10 || !scores || count!=6)return -1;
    for(unsigned i=0;i<6;i++)if(q10[i]<-1024 || q10[i]>1024)return -1;
    int32_t hidden[32];
    for(unsigned i=0;i<32;i++) {
        int64_t sum=behavior_bias0[i];
        for(unsigned j=0;j<6;j++)sum+=(int64_t)behavior_weight0[i][j]*q10[j];
        sum=down(sum,BEHAVIOR_SHIFT0);hidden[i]=sum<0 ? 0:sum>6144 ? 6144:(int32_t)sum;
    }
    unsigned best=0;
    for(unsigned i=0;i<4;i++) {
        int64_t sum=behavior_bias1[i];
        for(unsigned j=0;j<32;j++)sum+=(int64_t)behavior_weight1[i][j]*hidden[j];
        sum=down(sum,BEHAVIOR_SHIFT1);
        if(sum>INT32_MAX || sum<INT32_MIN)return -1;
        scores[i]=(int32_t)sum;if(scores[i]>scores[best])best=i;
    }
    return (int)best;
}
