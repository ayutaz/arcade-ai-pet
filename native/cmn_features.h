/* SPDX-License-Identifier: Apache-2.0 */
#ifndef PET_CMN_FEATURES_H
#define PET_CMN_FEATURES_H
#include "decision_features.h"
/* Trusted aligned caller-owned buffers, not a guest ABI. */
int pet_cmn_center(float *, size_t);
size_t pet_cmn_feature_scratch_size(void);
int pet_cmn_features(const int16_t *, size_t, float *, size_t, void *, size_t);
int pet_cmn_features_banked(const int16_t *const *, size_t, float *, size_t, void *, size_t);
#endif
