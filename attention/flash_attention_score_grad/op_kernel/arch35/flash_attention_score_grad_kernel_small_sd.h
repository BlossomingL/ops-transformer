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
    using SmallSDTilingData = FlashAttentionScoreGradSmallSDTilingData<IS_TND>;
    using SmallSDFagTilingData = typename SmallSDTilingData::FagTilingData;
    using SmallSDTilingType = const __gm__ SmallSDTilingData *__restrict;
    using SmallSDFagTilingType = const __gm__ SmallSDFagTilingData *__restrict;
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse,
                                GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR workspace,
                                SmallSDTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitSmallSDTndCursor();
    __aicore__ inline void SetSmallSDAxisRunInfo(FagRunInfo &runInfo, int64_t index);
    __aicore__ inline void UpdateSmallSDGmOffset(FagRunInfo &runInfo, int64_t index);
    __aicore__ inline void SetSmallSDRunInfo(FagRunInfo &runInfo, FagRunInfo &nextRunInfo, int64_t taskId,
                                             int64_t index, int64_t nextIndex);
    SmallSDTilingType smallSDTilingData;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink,
    GM_ADDR workspace, SmallSDTilingType ordTilingData, TPipe *pipeIn)
{
    smallSDTilingData = ordTilingData;
    SmallSDFagTilingType fagTilingData = &ordTilingData->fagTilingData;
    BaseKernel::Init(key, value, dy, query, pseShift, dropMask, attenMask, y, softmaxMax, softmaxSum, prefixN,
                     actualSeqQlen, actualSeqKvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope, keyRope,
                     sink, dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, fagTilingData, pipeIn);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDTndCursor()
{
    if constexpr (IS_TND) {
        const int64_t startBatch = smallSDTilingData->tndCoreParam[this->cBlockIdx].startBatch;
        const int64_t startN2 = smallSDTilingData->tndCoreParam[this->cBlockIdx].startN2;
        const int64_t blockStart = smallSDTilingData->tndCoreParam[this->cBlockIdx].blockStart;
        const int64_t qPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].qPrefix;
        const int64_t kvPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].kvPrefix;
        this->curBatchIdx = startBatch;
        this->curBatchTotalBaseIdx = blockStart - startN2;
        this->curBatchTotalS1BOffset = qPrefix * this->constInfo.commonConstInfo.n2GD;
        this->curBatchTotalS2BOffset = kvPrefix * this->constInfo.commonConstInfo.n2D;
        this->curBatchTotalS1BOffsetForDv = qPrefix * this->constInfo.commonConstInfo.n2GDv;
        this->curBatchTotalS2BOffsetForDv = kvPrefix * this->constInfo.commonConstInfo.n2Dv;
        this->curBatchTotalS1S2SizeAlign = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2AlignPrefix;
        this->curBatchTotalS1S2Size = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2Prefix;
        this->curBatchTotalS2Size = kvPrefix;
        if constexpr (IS_ROPE) {
            this->curBatchTotalS1BRopeOffset = qPrefix * this->constInfo.commonConstInfo.n2GDr;
            this->curBatchTotalS2BRopeOffset = kvPrefix * this->constInfo.commonConstInfo.n2Dr;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetSmallSDAxisRunInfo(FagRunInfo &runInfo,
                                                                                         int64_t index)
{
    const int64_t n2Size = smallSDTilingData->baseParam.n2Size;
    const int64_t boIdx = index / n2Size;
    const int64_t n2oIdx = index - boIdx * n2Size;
    int64_t actualS1Len = smallSDTilingData->baseParam.s1;
    int64_t actualS2Len = smallSDTilingData->baseParam.s2;
    if constexpr (IS_TND) {
        this->GetSeqQlenKvlenByBidx(boIdx, actualS1Len, actualS2Len);
    }
    this->SetAxisRunInfo(runInfo, 0, actualS2Len, boIdx, n2oIdx, 0, 0, 0);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::UpdateSmallSDGmOffset(FagRunInfo &runInfo,
                                                                                         int64_t index)
{
    if constexpr (IS_TND) {
        const int64_t n2oIdx = runInfo.commonRunInfo.n2oIdx;
        const int64_t qOffset = runInfo.lastBatchTotalS1BOffset + n2oIdx * this->constInfo.commonConstInfo.dSize;
        const int64_t kvOffset = runInfo.lastBatchTotalS2BOffset + n2oIdx * this->constInfo.commonConstInfo.dSize;
        runInfo.commonRunInfo.queryOffset = qOffset;
        runInfo.dyOffset = qOffset;
        runInfo.commonRunInfo.keyOffset = kvOffset;
        runInfo.commonRunInfo.valueOffset = kvOffset;
        runInfo.queryOffsetWithRope = qOffset;
        runInfo.keyOffsetWithRope = kvOffset;
        runInfo.queryOffsetWithRopeForMm12 = qOffset;
        runInfo.keyOffsetWithRopeForMm12 = kvOffset;
    } else {
        const int64_t n2Size = smallSDTilingData->baseParam.n2Size;
        const int64_t blockStart = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockStart;
        const int64_t startN2 = blockStart % n2Size;
        const int64_t taskOffset = index - blockStart;
        const int64_t batchStep = (startN2 + taskOffset) / n2Size;
        const int64_t groupStep = taskOffset - batchStep;
        const int64_t qOffset = smallSDTilingData->coreTaskParam[this->cBlockIdx].qOffset +
                                groupStep * smallSDTilingData->strideParam.qGroup +
                                batchStep * smallSDTilingData->strideParam.qS;
        const int64_t kvOffset = smallSDTilingData->coreTaskParam[this->cBlockIdx].kOffset +
                                 groupStep * smallSDTilingData->strideParam.kvGroup +
                                 batchStep * smallSDTilingData->strideParam.kvS;
        runInfo.commonRunInfo.queryOffset = qOffset;
        runInfo.dyOffset = qOffset;
        runInfo.commonRunInfo.keyOffset = kvOffset;
        runInfo.commonRunInfo.valueOffset = kvOffset;
        runInfo.queryOffsetWithRope = qOffset;
        runInfo.keyOffsetWithRope = kvOffset;
        runInfo.queryOffsetWithRopeForMm12 = qOffset;
        runInfo.keyOffsetWithRopeForMm12 = kvOffset;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetSmallSDRunInfo(FagRunInfo &runInfo,
                                                                                     FagRunInfo &nextRunInfo,
                                                                                     int64_t taskId, int64_t index,
                                                                                     int64_t nextIndex)
{
    SetSmallSDAxisRunInfo(runInfo, index);
    if (nextIndex != -1) {
        SetSmallSDAxisRunInfo(nextRunInfo, nextIndex);
    }
    this->SetRunInfo(runInfo, nextRunInfo, taskId, index, nextIndex);
    UpdateSmallSDGmOffset(runInfo, index);
    if (nextIndex != -1) {
        UpdateSmallSDGmOffset(nextRunInfo, nextIndex);
        this->preloadArgs.nextQueryOffset = nextRunInfo.commonRunInfo.queryOffset;
        this->preloadArgs.nextDyOffset = nextRunInfo.dyOffset;
        this->preloadArgs.nextMOrN = nextRunInfo.commonRunInfo.s1RealSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    static_assert(SPLIT_AXIS == BN2, "SmallSD only supports BN2 split axis.");
    static_assert(!IS_BN2_MULTIBLK, "SmallSD does not support BN2 multi block.");

    const int64_t groupCount = smallSDTilingData->coreTaskParam[this->cBlockIdx].groupCount;
    if (groupCount == 0) {
        return;
    }
    InitSmallSDTndCursor();

    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    int64_t nextValidBlockInnerIdx = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockStart;
    int64_t blockInnerIdx = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockStart;
    const int64_t blockEnd = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockEnd;

    FagRunInfo prevRunInfo;
    bool needSyncDkMM = false;
    while (true) {
        this->isLastLoop = (blockInnerIdx == -1);
        if (taskId > 0) {
            prevRunInfo = runInfos[(taskId + 1) & 1];
            this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
        }
        if (!this->isLastLoop) {
            nextValidBlockInnerIdx = (blockInnerIdx + 1 < blockEnd) ? (blockInnerIdx + 1) : -1;
            SetSmallSDRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx,
                              nextValidBlockInnerIdx);

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
