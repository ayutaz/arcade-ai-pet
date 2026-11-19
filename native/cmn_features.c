/* SPDX-License-Identifier: Apache-2.0 */
#include "cmn_features.h"
#include <math.h>

int pet_cmn_center(float *output, size_t count)
{
    if (!output || count != PET_DECISION_FEATURE_COUNT) return -1;
    /* Validate the complete input before changing any value. */
    for (size_t i=0; i<count; i++)
        if (!isfinite(output[i]) || fabsf(output[i])>16) return -1;
    if (output[260]<0 || output[260]>4) return -1;
    for (unsigned coefficient=0; coefficient<12; coefficient++) {
        double sum=0;
        for (unsigned bin=0; bin<20; bin++) sum+=(double)output[bin*12+coefficient];
        double mean=sum/20.0;
        for (unsigned bin=0; bin<20; bin++) {
            unsigned i=bin*12+coefficient;
            output[i]=(float)fmin(16.0, fmax(-16.0, (double)output[i]-mean));
        }
    }
    /* Energy[240:260] and duration[260] retain their original bits. */
    return 1;
}

size_t pet_cmn_feature_scratch_size(void) { return pet_decision_feature_scratch_size(); }

int pet_cmn_features(const int16_t *pcm, size_t samples, float *output, size_t count,
                     void *workspace, size_t workspace_bytes)
{
    int status=pet_decision_features(pcm,samples,output,count,workspace,workspace_bytes);
    return status==1 ? pet_cmn_center(output,count) : status;
}

int pet_cmn_features_banked(const int16_t *const *banks, size_t samples, float *output, size_t count,
                            void *workspace, size_t workspace_bytes)
{
    int status=pet_decision_features_banked(banks,samples,output,count,workspace,workspace_bytes);
    return status==1 ? pet_cmn_center(output,count) : status;
}
