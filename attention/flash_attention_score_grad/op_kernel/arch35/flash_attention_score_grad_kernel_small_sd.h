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

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_SMALL_SD_H

#include "flash_attention_score_grad_kernel.h"

namespace FagBaseApi {

template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelSmallSD : public FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseKernel = FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>;
    using KernelBaseClass = typename BaseKernel::BaseClass;
    __aicore__ inline void Process();
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    static_assert(SPLIT_AXIS == BN2, "SmallSD only supports BN2 split axis.");
    static_assert(!IS_BN2_MULTIBLK, "SmallSD does not support BN2 multi block.");

    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    nextValidBlockInnerIdx = this->GetNextValidIdx(
        runInfos[0], taskId, this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
    blockInnerIdx = nextValidBlockInnerIdx;

    FagRunInfo prevRunInfo;
    bool needSyncDkMM = false;
    while (true) {
        this->isLastLoop = (blockInnerIdx == -1);
        if (taskId > 0) {
            prevRunInfo = runInfos[(taskId + 1) & 1];
            this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
        }
        if (!this->isLastLoop) {
            nextValidBlockInnerIdx =
                this->GetNextValidIdx(runInfos[(taskId + 1) & 1], taskId + 1, blockInnerIdx + 1, curLoopIdx + 1);
            this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx,
                             nextValidBlockInnerIdx);
            if (this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx || IS_TND_SWIZZLE) {
                curLoopIdx++;
            } else {
                blockInnerIdx++;
            }

            if constexpr (KernelBaseClass::IS_DK_WRITE_UB) {
                if ASCEND_IS_AIC {
                    if (needSyncDkMM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                    }
                }
            }

            LocalTensor<CALC_TYPE> mm2ResTensor =
                this->mm2ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
            }

            LocalTensor<CALC_TYPE> mm1ResTensor =
                this->mm1ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
            }

            this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId & 1], taskId);
        }
        if (taskId > 0) {
            this->ComputeDqkvBn2(prevRunInfo, needSyncDkMM, taskId);
        }
        if (blockInnerIdx == -1) {
            break;
        }
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}

} // namespace FagBaseApi
#endif
