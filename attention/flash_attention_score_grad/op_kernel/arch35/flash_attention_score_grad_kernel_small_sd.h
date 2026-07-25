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
 * \file flash_attention_score_grad_kernel_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H_
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H_

#include "flash_attention_score_grad_block_cube_small_sd.h"
#include "flash_attention_score_grad_block_vec_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"

namespace FagSmallSDApi {
struct SmallSDPipelineSlot {
    uint32_t taskId;
    uint32_t slotId;
    uint32_t batchIdx;
    uint32_t n2Idx;
    uint16_t actualS1;
    uint16_t actualS2;
    SmallSDCoreTaskParamRegbase offsets;
};

template <typename CubeBlock, typename VectorBlock, bool IsTnd, uint32_t HeadDim, uint32_t Layout, bool IsSingleTask>
class FlashAttentionScoreGradKernelSmallSD {
public:
    __aicore__ inline void Init(const __gm__ FlashAttentionScoreGradSmallSDTilingData *tilingData, TPipe *pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        cubeBlock_.Init(pipe_);
        vectorBlock_.Init(pipe_);
    }

    __aicore__ inline void Process()
    {
        if (tilingData_ == nullptr) {
            return;
        }
        const SmallSDBaseParamRegbase base = tilingData_->baseParam;
        const uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= base.usedCoreNum) {
            return;
        }
        const SmallSDCoreTaskParamRegbase core = tilingData_->coreTaskParam[coreIdx];
        if (core.groupCount == 0) {
            return;
        }
        if constexpr (IsSingleTask) {
            ProcessOneGroup(core, base);
        } else {
            for (uint32_t task = 0; task < core.groupCount; ++task) {
                ProcessOneGroup(core, base);
            }
        }
    }

private:
    __aicore__ inline void ProcessOneGroup(const SmallSDCoreTaskParamRegbase &core,
                                           const SmallSDBaseParamRegbase &base)
    {
        cubeBlock_.IssueFirstStage(core, base);
        vectorBlock_.ConsumeFirstStage(core, base);
        cubeBlock_.IssueGradStage(core, base);
        vectorBlock_.CommitOutputs(core, base);
    }

    const __gm__ FlashAttentionScoreGradSmallSDTilingData *tilingData_ = nullptr;
    TPipe *pipe_ = nullptr;
    CubeBlock cubeBlock_;
    VectorBlock vectorBlock_;
};
} // namespace FagSmallSDApi

#endif // FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H_
