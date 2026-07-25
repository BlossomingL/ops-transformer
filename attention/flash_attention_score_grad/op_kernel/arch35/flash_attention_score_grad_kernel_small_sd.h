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

#include "flash_attention_score_grad_common_small_sd.h"
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
    __aicore__ inline void InitSmallSDConstInfo();
    __aicore__ inline void InitSmallSDCursor();
    __aicore__ inline void LoadSmallSDTndBatch();
    __aicore__ inline void PrepareSmallSDSlot(SmallSDPipelineSlot &slot, int64_t taskId);
    __aicore__ inline void AdvanceSmallSDCursor();
    __aicore__ inline void BuildCompatRunInfo(FagRunInfo &runInfo, const SmallSDPipelineSlot &slot);
    __aicore__ inline void SetSmallSDPreloadArgs(FagRunInfo &runInfo, const SmallSDPipelineSlot &nextSlot,
                                                 bool hasNext);
    SmallSDTilingType smallSDTilingData;
    SmallSDConstInfo smallSDConstInfo;
    SmallSDTaskCursor smallSDCursor;
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
    InitSmallSDConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDConstInfo()
{
    smallSDConstInfo.bSize = smallSDTilingData->baseParam.bSize;
    smallSDConstInfo.n1Size = smallSDTilingData->baseParam.n1Size;
    smallSDConstInfo.n2Size = smallSDTilingData->baseParam.n2Size;
    smallSDConstInfo.gSize = smallSDTilingData->baseParam.gSize;
    smallSDConstInfo.d = smallSDTilingData->baseParam.actualD;
    smallSDConstInfo.dv = smallSDTilingData->baseParam.actualDv;
    smallSDConstInfo.dAlign16 = AlignTo16(smallSDConstInfo.d);
    smallSDConstInfo.dvAlign16 = AlignTo16(smallSDConstInfo.dv);
    smallSDConstInfo.layoutType = smallSDTilingData->baseParam.layoutType;
    smallSDConstInfo.tndMaxSumLayout = smallSDTilingData->baseParam.tndMaxSumLayout;
    smallSDConstInfo.isSingleTask = smallSDTilingData->baseParam.isSingleTask;
    smallSDConstInfo.blockStart = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockStart;
    smallSDConstInfo.blockEnd = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockEnd;
    smallSDConstInfo.groupCount = smallSDTilingData->coreTaskParam[this->cBlockIdx].groupCount;
    smallSDConstInfo.scaleValue = smallSDTilingData->baseParam.scaleValue;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::LoadSmallSDTndBatch()
{
    if constexpr (IS_TND) {
        smallSDCursor.qEnd = ((__gm__ int64_t *)this->actualSeqQlenAddr)[smallSDCursor.batchIdx];
        smallSDCursor.kvEnd = ((__gm__ int64_t *)this->actualSeqKvlenAddr)[smallSDCursor.batchIdx];
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDCursor()
{
    smallSDCursor = {};
    if constexpr (IS_TND) {
        smallSDCursor.batchIdx = smallSDTilingData->tndCoreParam[this->cBlockIdx].startBatch;
        smallSDCursor.n2Idx = smallSDTilingData->tndCoreParam[this->cBlockIdx].startN2;
        smallSDCursor.qPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].qPrefix;
        smallSDCursor.kvPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].kvPrefix;
        smallSDCursor.s1s2Prefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2Prefix;
        smallSDCursor.s1s2AlignPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2AlignPrefix;
        LoadSmallSDTndBatch();
    } else {
        const int64_t blockStart = smallSDConstInfo.blockStart;
        smallSDCursor.batchIdx = blockStart / smallSDConstInfo.n2Size;
        smallSDCursor.n2Idx = blockStart - smallSDCursor.batchIdx * smallSDConstInfo.n2Size;
        smallSDCursor.qPrefix = smallSDCursor.batchIdx * smallSDTilingData->baseParam.s1;
        smallSDCursor.kvPrefix = smallSDCursor.batchIdx * smallSDTilingData->baseParam.s2;
        smallSDCursor.s1s2Prefix =
            smallSDCursor.batchIdx * smallSDTilingData->baseParam.s1 * smallSDTilingData->baseParam.s2;
        smallSDCursor.s1s2AlignPrefix =
            smallSDCursor.batchIdx * smallSDTilingData->baseParam.s1 * smallSDTilingData->baseParam.s2Align16;
        smallSDCursor.offsets.q = smallSDTilingData->coreTaskParam[this->cBlockIdx].qOffset;
        smallSDCursor.offsets.k = smallSDTilingData->coreTaskParam[this->cBlockIdx].kOffset;
        smallSDCursor.offsets.v = smallSDTilingData->coreTaskParam[this->cBlockIdx].vOffset;
        smallSDCursor.offsets.dy = smallSDTilingData->coreTaskParam[this->cBlockIdx].dyOffset;
        smallSDCursor.offsets.attention = smallSDTilingData->coreTaskParam[this->cBlockIdx].attentionOffset;
        smallSDCursor.offsets.softmaxMax = smallSDTilingData->coreTaskParam[this->cBlockIdx].maxOffset;
        smallSDCursor.offsets.softmaxSum = smallSDTilingData->coreTaskParam[this->cBlockIdx].sumOffset;
        smallSDCursor.offsets.dq = smallSDTilingData->coreTaskParam[this->cBlockIdx].dqOffset;
        smallSDCursor.offsets.dk = smallSDTilingData->coreTaskParam[this->cBlockIdx].dkOffset;
        smallSDCursor.offsets.dv = smallSDTilingData->coreTaskParam[this->cBlockIdx].dvOffset;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::PrepareSmallSDSlot(SmallSDPipelineSlot &slot,
                                                                                      int64_t taskId)
{
    slot = {};
    slot.taskId = taskId;
    slot.batchIdx = smallSDCursor.batchIdx;
    slot.n2Idx = smallSDCursor.n2Idx;
    slot.qPrefix = smallSDCursor.qPrefix;
    slot.kvPrefix = smallSDCursor.kvPrefix;
    slot.s1s2Prefix = smallSDCursor.s1s2Prefix;
    slot.s1s2AlignPrefix = smallSDCursor.s1s2AlignPrefix;
    slot.shape.d = smallSDConstInfo.d;
    slot.shape.dv = smallSDConstInfo.dv;
    slot.shape.scale = smallSDConstInfo.scaleValue;
    if constexpr (IS_TND) {
        slot.shape.s1 = smallSDCursor.qEnd - smallSDCursor.qPrefix;
        slot.shape.s2 = smallSDCursor.kvEnd - smallSDCursor.kvPrefix;
        slot.shape.s2Align16 = AlignTo16(slot.shape.s2);
        const int64_t n2Offset = smallSDCursor.n2Idx * smallSDConstInfo.d;
        slot.offsets.q = smallSDCursor.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d + n2Offset;
        slot.offsets.k = smallSDCursor.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d + n2Offset;
        slot.offsets.v = slot.offsets.k;
        slot.offsets.dy = slot.offsets.q;
        slot.offsets.attention = slot.offsets.q;
        slot.offsets.dq = slot.offsets.q;
        slot.offsets.dk = slot.offsets.k;
        slot.offsets.dv = slot.offsets.k;
        slot.offsets.softmaxMax = 0;
        slot.offsets.softmaxSum = 0;
    } else {
        slot.shape.s1 = smallSDTilingData->baseParam.s1;
        slot.shape.s2 = smallSDTilingData->baseParam.s2;
        slot.shape.s2Align16 = smallSDTilingData->baseParam.s2Align16;
        slot.offsets = smallSDCursor.offsets;
    }
    slot.shape.halfS1 = (slot.shape.s1 + 1) >> 1;
    slot.shape.firstHalfS1 = slot.shape.halfS1;
    slot.shape.halfS2 = (slot.shape.s2 + 1) >> 1;
    slot.shape.firstHalfS2 = slot.shape.halfS2;
    if (this->vSubBlockIdx == 1) {
        slot.shape.halfS1 = slot.shape.s1 - slot.shape.halfS1;
        slot.shape.halfS2 = slot.shape.s2 - slot.shape.halfS2;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceSmallSDCursor()
{
    if (smallSDCursor.n2Idx + 1 < smallSDConstInfo.n2Size) {
        smallSDCursor.n2Idx++;
        if constexpr (IS_TND) {
            const int64_t dOffset = smallSDConstInfo.d;
            smallSDCursor.offsets.q += dOffset;
            smallSDCursor.offsets.k += dOffset;
            smallSDCursor.offsets.v += dOffset;
            smallSDCursor.offsets.dy += dOffset;
            smallSDCursor.offsets.dq += dOffset;
            smallSDCursor.offsets.dk += dOffset;
            smallSDCursor.offsets.dv += dOffset;
        } else {
            smallSDCursor.offsets.q += smallSDTilingData->strideParam.qGroup;
            smallSDCursor.offsets.k += smallSDTilingData->strideParam.kvGroup;
            smallSDCursor.offsets.v += smallSDTilingData->strideParam.kvGroup;
            smallSDCursor.offsets.dy += smallSDTilingData->strideParam.dyGroup;
            smallSDCursor.offsets.attention += smallSDTilingData->strideParam.attentionGroup;
            smallSDCursor.offsets.dq += smallSDTilingData->strideParam.dqGroup;
            smallSDCursor.offsets.dk += smallSDTilingData->strideParam.dkvGroup;
            smallSDCursor.offsets.dv += smallSDTilingData->strideParam.dkvGroup;
        }
        return;
    }

    smallSDCursor.n2Idx = 0;
    smallSDCursor.batchIdx++;
    if constexpr (IS_TND) {
        const int64_t s1 = smallSDCursor.qEnd - smallSDCursor.qPrefix;
        const int64_t s2 = smallSDCursor.kvEnd - smallSDCursor.kvPrefix;
        smallSDCursor.qPrefix = smallSDCursor.qEnd;
        smallSDCursor.kvPrefix = smallSDCursor.kvEnd;
        smallSDCursor.s1s2Prefix += s1 * s2;
        smallSDCursor.s1s2AlignPrefix += s1 * AlignTo16(s2);
        LoadSmallSDTndBatch();
    } else {
        smallSDCursor.qPrefix += smallSDTilingData->baseParam.s1;
        smallSDCursor.kvPrefix += smallSDTilingData->baseParam.s2;
        smallSDCursor.s1s2Prefix += smallSDTilingData->baseParam.s1 * smallSDTilingData->baseParam.s2;
        smallSDCursor.s1s2AlignPrefix +=
            smallSDTilingData->baseParam.s1 * smallSDTilingData->baseParam.s2Align16;
        smallSDCursor.offsets.q += smallSDTilingData->strideParam.qS;
        smallSDCursor.offsets.k += smallSDTilingData->strideParam.kvS;
        smallSDCursor.offsets.v += smallSDTilingData->strideParam.kvS;
        smallSDCursor.offsets.dy += smallSDTilingData->strideParam.qS;
        smallSDCursor.offsets.attention += smallSDTilingData->strideParam.qS;
        smallSDCursor.offsets.dq += smallSDTilingData->strideParam.qS;
        smallSDCursor.offsets.dk += smallSDTilingData->strideParam.kvS;
        smallSDCursor.offsets.dv += smallSDTilingData->strideParam.kvS;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::BuildCompatRunInfo(FagRunInfo &runInfo,
                                                                                      const SmallSDPipelineSlot &slot)
{
    runInfo = {};
    runInfo.commonRunInfo.boIdx = slot.batchIdx;
    runInfo.commonRunInfo.n2oIdx = slot.n2Idx;
    runInfo.commonRunInfo.goIdx = 0;
    runInfo.commonRunInfo.s1oIdx = 0;
    runInfo.commonRunInfo.taskId = slot.taskId;
    runInfo.commonRunInfo.taskIdMod2 = slot.taskId & 1;
    runInfo.commonRunInfo.s1RealSize = slot.shape.s1;
    runInfo.commonRunInfo.s2RealSize = slot.shape.s2;
    runInfo.commonRunInfo.actualS1Size = slot.shape.s1;
    runInfo.commonRunInfo.actualS2Size = slot.shape.s2;
    runInfo.commonRunInfo.halfS1RealSize = slot.shape.halfS1;
    runInfo.commonRunInfo.firstHalfS1RealSize = slot.shape.firstHalfS1;
    runInfo.commonRunInfo.s2SizeAcc = slot.kvPrefix;
    runInfo.commonRunInfo.b1SSOffsetAlign = slot.s1s2AlignPrefix;
    runInfo.commonRunInfo.b1SSOffset = slot.s1s2Prefix;
    runInfo.commonRunInfo.b1SSAttenMaskOffset = slot.s1s2Prefix;
    runInfo.commonRunInfo.s2StartIdx = 0;
    runInfo.commonRunInfo.s2AlignedSize = slot.shape.s2Align16;
    runInfo.commonRunInfo.vecCoreOffset = this->vSubBlockIdx * slot.shape.firstHalfS1;
    runInfo.commonRunInfo.queryOffset = slot.offsets.q;
    runInfo.commonRunInfo.keyOffset = slot.offsets.k;
    runInfo.commonRunInfo.valueOffset = slot.offsets.v;
    runInfo.s2oIdx = 0;
    runInfo.s2CvBegin = 0;
    runInfo.s2CvEnd = slot.shape.s2;
    runInfo.halfS2RealSize = slot.shape.halfS2;
    runInfo.firstHalfS2RealSize = slot.shape.firstHalfS2;
    runInfo.dyOffset = slot.offsets.dy;
    runInfo.queryOffsetWithRope = slot.offsets.q;
    runInfo.keyOffsetWithRope = slot.offsets.k;
    runInfo.queryOffsetWithRopeForMm12 = slot.offsets.q;
    runInfo.keyOffsetWithRopeForMm12 = slot.offsets.k;
    runInfo.maxsumOffset = slot.offsets.softmaxMax;
    runInfo.lastBatchIdx = slot.batchIdx;
    runInfo.lastBatchTotalS1BOffset = slot.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d;
    runInfo.lastBatchTotalS2BOffset = slot.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d;
    runInfo.lastBatchTotalS1BOffsetForDv = slot.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.dv;
    runInfo.lastBatchTotalS2BOffsetForDv = slot.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.dv;
    runInfo.lastBatchTotalS1S2SizeAlign = slot.s1s2AlignPrefix;
    runInfo.lastBatchTotalS1S2Size = slot.s1s2Prefix;
    runInfo.lastBatchTotalS2Size = slot.kvPrefix;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetSmallSDPreloadArgs(
    FagRunInfo &runInfo, const SmallSDPipelineSlot &nextSlot, bool hasNext)
{
    this->preloadArgs.copyNext = hasNext;
    this->preloadArgs.copyCurrent = runInfo.commonRunInfo.taskId == 0;
    runInfo.isNextS2IdxNoChange = false;
    if (!hasNext) {
        return;
    }
    this->preloadArgs.nextQueryOffset = nextSlot.offsets.q;
    this->preloadArgs.nextDyOffset = nextSlot.offsets.dy;
    this->preloadArgs.nextMOrN = nextSlot.shape.s1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    static_assert(SPLIT_AXIS == BN2, "SmallSD only supports BN2 split axis.");
    static_assert(!IS_BN2_MULTIBLK, "SmallSD does not support BN2 multi block.");

    const int64_t groupCount = smallSDConstInfo.groupCount;
    if (groupCount == 0) {
        return;
    }
    InitSmallSDCursor();

    SmallSDPipelineSlot slots[2];
    SmallSDPipelineSlot nextSlot;
    FagRunInfo compatRunInfos[2]; // temporary adapter for current BN2 cube/vector block APIs
    FagRunInfo prevRunInfo;
    bool needSyncDkMM = false;
    for (int64_t taskId = 0; taskId <= groupCount; ++taskId) {
        this->isLastLoop = taskId == groupCount;
        if (taskId > 0) {
            prevRunInfo = compatRunInfos[(taskId + 1) & 1];
            this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
        }
        if (!this->isLastLoop) {
            PrepareSmallSDSlot(slots[taskId & 1], taskId);
            const bool hasNext = (taskId + 1 < groupCount);
            if (hasNext) {
                AdvanceSmallSDCursor();
                PrepareSmallSDSlot(nextSlot, taskId + 1);
            }
            BuildCompatRunInfo(compatRunInfos[taskId & 1], slots[taskId & 1]);
            SetSmallSDPreloadArgs(compatRunInfos[taskId & 1], nextSlot, hasNext);

            if constexpr (KernelBaseClass::IS_DK_WRITE_UB) {
                if ASCEND_IS_AIC {
                    if (needSyncDkMM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                    }
                }
            }

            LocalTensor<CALC_TYPE> mm2ResTensor =
                this->mm2ResBuf[compatRunInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, compatRunInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
            }

            LocalTensor<CALC_TYPE> mm1ResTensor =
                this->mm1ResBuf[compatRunInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, compatRunInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
            }

            this->vecBlock.CopyMaxSum(this->constInfo, compatRunInfos[taskId & 1], taskId);
        }
        if (taskId > 0) {
            this->ComputeDqkvBn2(prevRunInfo, needSyncDkMM, taskId);
        }
    }
}

} // namespace FagBaseApi
#endif
