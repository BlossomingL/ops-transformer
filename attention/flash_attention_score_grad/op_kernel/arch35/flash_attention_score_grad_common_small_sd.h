/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_common_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_COMMON_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_COMMON_SMALL_SD_H

#include "flash_attention_score_grad_tiling_data_regbase.h"

namespace FagBaseApi {

struct SmallSDShape {
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dv;
    int64_t s2Align16;
    int64_t halfS1;
    int64_t firstHalfS1;
    int64_t halfS2;
    int64_t firstHalfS2;
    float scale;
};

struct SmallSDOffsets {
    int64_t q;
    int64_t k;
    int64_t v;
    int64_t dy;
    int64_t attention;
    int64_t softmaxMax;
    int64_t softmaxSum;
    int64_t dq;
    int64_t dk;
    int64_t dv;
};

struct SmallSDConstInfo {
    int64_t bSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t gSize;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dv;
    int64_t s2Align16;
    int64_t dAlign16;
    int64_t dvAlign16;
    uint64_t qGroupStride;
    uint64_t kvGroupStride;
    uint64_t dyGroupStride;
    uint64_t attentionGroupStride;
    uint64_t dqGroupStride;
    uint64_t dkvGroupStride;
    uint64_t qSStride;
    uint64_t kvSStride;
    uint32_t layoutType;
    uint32_t tndMaxSumLayout;
    uint32_t isSingleTask;
    uint32_t blockStart;
    uint32_t blockEnd;
    uint32_t groupCount;
    float scaleValue;
    int64_t startBatch;
    int64_t startN2;
    int64_t startQPrefix;
    int64_t startKvPrefix;
    int64_t startS1S2Prefix;
    int64_t startS1S2AlignPrefix;
    SmallSDOffsets initialOffsets;
};

struct SmallSDTaskCursor {
    int64_t batchIdx;
    int64_t n2Idx;
    int64_t qPrefix;
    int64_t kvPrefix;
    int64_t qEnd;
    int64_t kvEnd;
    int64_t s1s2Prefix;
    int64_t s1s2AlignPrefix;
    SmallSDOffsets offsets;
};

struct SmallSDRunInfo {
    int64_t taskId;
    int64_t batchIdx;
    int64_t n2Idx;
    int64_t qPrefix;
    int64_t kvPrefix;
    int64_t s1s2Prefix;
    int64_t s1s2AlignPrefix;
    SmallSDShape shape;
    SmallSDOffsets offsets;
};

} // namespace FagBaseApi
#endif
