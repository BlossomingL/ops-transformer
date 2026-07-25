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
    __aicore__ inline void PrepareSmallSDRunInfo(SmallSDRunInfo &runInfo, int64_t taskId);
    __aicore__ inline void AdvanceSmallSDCursor();
    __aicore__ inline void SetSmallSDPreloadArgs(const SmallSDRunInfo &runInfo, const SmallSDRunInfo &nextRunInfo,
                                                bool hasNext);
    __aicore__ inline void ProcessVec1SmallSD(const SmallSDRunInfo &runInfo);
    __aicore__ inline void IssueMm1Mm2SmallSD(const SmallSDRunInfo &runInfo, const SmallSDRunInfo &nextRunInfo,
                                             bool hasNext, bool &needSyncDkMM);
    __aicore__ inline void ComputeDqkvSmallSD(const SmallSDRunInfo &runInfo, bool &needSyncDkMM, int64_t taskId);
    __aicore__ inline void ProcessSingleGroupSmallSD();
    __aicore__ inline void ProcessMultiGroupSmallSD(int64_t groupCount);
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
    this->vecBlock.SetSmallSDCompatConstInfo(this->constInfo);
    InitSmallSDConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::InitSmallSDConstInfo()
{
    smallSDConstInfo = {};
    smallSDConstInfo.bSize = smallSDTilingData->baseParam.bSize;
    smallSDConstInfo.n1Size = smallSDTilingData->baseParam.n1Size;
    smallSDConstInfo.n2Size = smallSDTilingData->baseParam.n2Size;
    smallSDConstInfo.gSize = smallSDTilingData->baseParam.gSize;
    smallSDConstInfo.s1 = smallSDTilingData->baseParam.s1;
    smallSDConstInfo.s2 = smallSDTilingData->baseParam.s2;
    smallSDConstInfo.d = smallSDTilingData->baseParam.actualD;
    smallSDConstInfo.dv = smallSDTilingData->baseParam.actualDv;
    smallSDConstInfo.s2Align16 = smallSDTilingData->baseParam.s2Align16;
    smallSDConstInfo.dAlign16 = AlignTo16(smallSDConstInfo.d);
    smallSDConstInfo.dvAlign16 = AlignTo16(smallSDConstInfo.dv);
    smallSDConstInfo.qGroupStride = smallSDTilingData->strideParam.qGroup;
    smallSDConstInfo.kvGroupStride = smallSDTilingData->strideParam.kvGroup;
    smallSDConstInfo.dyGroupStride = smallSDTilingData->strideParam.dyGroup;
    smallSDConstInfo.attentionGroupStride = smallSDTilingData->strideParam.attentionGroup;
    smallSDConstInfo.dqGroupStride = smallSDTilingData->strideParam.dqGroup;
    smallSDConstInfo.dkvGroupStride = smallSDTilingData->strideParam.dkvGroup;
    smallSDConstInfo.qSStride = smallSDTilingData->strideParam.qS;
    smallSDConstInfo.kvSStride = smallSDTilingData->strideParam.kvS;
    smallSDConstInfo.layoutType = smallSDTilingData->baseParam.layoutType;
    if constexpr (IS_TND) {
        smallSDConstInfo.qRowStride = smallSDConstInfo.n2Size * smallSDConstInfo.d;
        smallSDConstInfo.kvRowStride = smallSDConstInfo.n2Size * smallSDConstInfo.d;
    } else {
        if (smallSDConstInfo.layoutType == BNGSD) {
            smallSDConstInfo.qRowStride = smallSDConstInfo.d;
            smallSDConstInfo.kvRowStride = smallSDConstInfo.d;
        } else if (smallSDConstInfo.layoutType == SBNGD) {
            smallSDConstInfo.qRowStride = smallSDConstInfo.bSize * smallSDConstInfo.n2Size * smallSDConstInfo.d;
            smallSDConstInfo.kvRowStride = smallSDConstInfo.bSize * smallSDConstInfo.n2Size * smallSDConstInfo.d;
        } else {
            smallSDConstInfo.qRowStride = smallSDConstInfo.n2Size * smallSDConstInfo.d;
            smallSDConstInfo.kvRowStride = smallSDConstInfo.n2Size * smallSDConstInfo.d;
        }
    }
    smallSDConstInfo.tndMaxSumLayout = smallSDTilingData->baseParam.tndMaxSumLayout;
    smallSDConstInfo.isSingleTask = smallSDTilingData->baseParam.isSingleTask;
    smallSDConstInfo.blockStart = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockStart;
    smallSDConstInfo.blockEnd = smallSDTilingData->coreTaskParam[this->cBlockIdx].blockEnd;
    smallSDConstInfo.groupCount = smallSDTilingData->coreTaskParam[this->cBlockIdx].groupCount;
    smallSDConstInfo.scaleValue = smallSDTilingData->baseParam.scaleValue;
    smallSDConstInfo.initialOffsets.q = smallSDTilingData->coreTaskParam[this->cBlockIdx].qOffset;
    smallSDConstInfo.initialOffsets.k = smallSDTilingData->coreTaskParam[this->cBlockIdx].kOffset;
    smallSDConstInfo.initialOffsets.v = smallSDTilingData->coreTaskParam[this->cBlockIdx].vOffset;
    smallSDConstInfo.initialOffsets.dy = smallSDTilingData->coreTaskParam[this->cBlockIdx].dyOffset;
    smallSDConstInfo.initialOffsets.attention = smallSDTilingData->coreTaskParam[this->cBlockIdx].attentionOffset;
    smallSDConstInfo.initialOffsets.softmaxMax = smallSDTilingData->coreTaskParam[this->cBlockIdx].maxOffset;
    smallSDConstInfo.initialOffsets.softmaxSum = smallSDTilingData->coreTaskParam[this->cBlockIdx].sumOffset;
    smallSDConstInfo.initialOffsets.dq = smallSDTilingData->coreTaskParam[this->cBlockIdx].dqOffset;
    smallSDConstInfo.initialOffsets.dk = smallSDTilingData->coreTaskParam[this->cBlockIdx].dkOffset;
    smallSDConstInfo.initialOffsets.dv = smallSDTilingData->coreTaskParam[this->cBlockIdx].dvOffset;
    if constexpr (IS_TND) {
        smallSDConstInfo.startBatch = smallSDTilingData->tndCoreParam[this->cBlockIdx].startBatch;
        smallSDConstInfo.startN2 = smallSDTilingData->tndCoreParam[this->cBlockIdx].startN2;
        smallSDConstInfo.startQPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].qPrefix;
        smallSDConstInfo.startKvPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].kvPrefix;
        smallSDConstInfo.startS1S2Prefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2Prefix;
        smallSDConstInfo.startS1S2AlignPrefix = smallSDTilingData->tndCoreParam[this->cBlockIdx].s1s2AlignPrefix;
    }
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
        smallSDCursor.batchIdx = smallSDConstInfo.startBatch;
        smallSDCursor.n2Idx = smallSDConstInfo.startN2;
        smallSDCursor.qPrefix = smallSDConstInfo.startQPrefix;
        smallSDCursor.kvPrefix = smallSDConstInfo.startKvPrefix;
        smallSDCursor.s1s2Prefix = smallSDConstInfo.startS1S2Prefix;
        smallSDCursor.s1s2AlignPrefix = smallSDConstInfo.startS1S2AlignPrefix;
        LoadSmallSDTndBatch();
    } else {
        const int64_t blockStart = smallSDConstInfo.blockStart;
        smallSDCursor.batchIdx = blockStart / smallSDConstInfo.n2Size;
        smallSDCursor.n2Idx = blockStart - smallSDCursor.batchIdx * smallSDConstInfo.n2Size;
        smallSDCursor.qPrefix = smallSDCursor.batchIdx * smallSDConstInfo.s1;
        smallSDCursor.kvPrefix = smallSDCursor.batchIdx * smallSDConstInfo.s2;
        smallSDCursor.s1s2Prefix = smallSDCursor.batchIdx * smallSDConstInfo.s1 * smallSDConstInfo.s2;
        smallSDCursor.s1s2AlignPrefix = smallSDCursor.batchIdx * smallSDConstInfo.s1 * smallSDConstInfo.s2Align16;
        smallSDCursor.offsets = smallSDConstInfo.initialOffsets;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::PrepareSmallSDRunInfo(SmallSDRunInfo &runInfo,
                                                                                         int64_t taskId)
{
    runInfo = {};
    runInfo.taskId = taskId;
    runInfo.batchIdx = smallSDCursor.batchIdx;
    runInfo.n2Idx = smallSDCursor.n2Idx;
    runInfo.qPrefix = smallSDCursor.qPrefix;
    runInfo.kvPrefix = smallSDCursor.kvPrefix;
    runInfo.s1s2Prefix = smallSDCursor.s1s2Prefix;
    runInfo.s1s2AlignPrefix = smallSDCursor.s1s2AlignPrefix;
    runInfo.shape.d = smallSDConstInfo.d;
    runInfo.shape.dv = smallSDConstInfo.dv;
    runInfo.shape.scale = smallSDConstInfo.scaleValue;
    if constexpr (IS_TND) {
        runInfo.shape.s1 = smallSDCursor.qEnd - smallSDCursor.qPrefix;
        runInfo.shape.s2 = smallSDCursor.kvEnd - smallSDCursor.kvPrefix;
        runInfo.shape.s2Align16 = AlignTo16(runInfo.shape.s2);
        const int64_t n2Offset = smallSDCursor.n2Idx * smallSDConstInfo.d;
        runInfo.offsets.q = smallSDCursor.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d + n2Offset;
        runInfo.offsets.k = smallSDCursor.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d + n2Offset;
        runInfo.offsets.v = runInfo.offsets.k;
        runInfo.offsets.dy = runInfo.offsets.q;
        runInfo.offsets.attention = runInfo.offsets.q;
        runInfo.offsets.dq = runInfo.offsets.q;
        runInfo.offsets.dk = runInfo.offsets.k;
        runInfo.offsets.dv = runInfo.offsets.k;
        runInfo.offsets.softmaxMax = 0;
        runInfo.offsets.softmaxSum = 0;
    } else {
        runInfo.shape.s1 = smallSDConstInfo.s1;
        runInfo.shape.s2 = smallSDConstInfo.s2;
        runInfo.shape.s2Align16 = smallSDConstInfo.s2Align16;
        runInfo.offsets = smallSDCursor.offsets;
    }
    runInfo.shape.halfS1 = (runInfo.shape.s1 + 1) >> 1;
    runInfo.shape.firstHalfS1 = runInfo.shape.halfS1;
    runInfo.shape.halfS2 = (runInfo.shape.s2 + 1) >> 1;
    runInfo.shape.firstHalfS2 = runInfo.shape.halfS2;
    if (this->vSubBlockIdx == 1) {
        runInfo.shape.halfS1 = runInfo.shape.s1 - runInfo.shape.halfS1;
        runInfo.shape.halfS2 = runInfo.shape.s2 - runInfo.shape.halfS2;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::AdvanceSmallSDCursor()
{
    if (smallSDCursor.n2Idx + 1 < smallSDConstInfo.n2Size) {
        smallSDCursor.n2Idx++;
        if constexpr (!IS_TND) {
            smallSDCursor.offsets.q += smallSDConstInfo.qGroupStride;
            smallSDCursor.offsets.k += smallSDConstInfo.kvGroupStride;
            smallSDCursor.offsets.v += smallSDConstInfo.kvGroupStride;
            smallSDCursor.offsets.dy += smallSDConstInfo.dyGroupStride;
            smallSDCursor.offsets.attention += smallSDConstInfo.attentionGroupStride;
            smallSDCursor.offsets.dq += smallSDConstInfo.dqGroupStride;
            smallSDCursor.offsets.dk += smallSDConstInfo.dkvGroupStride;
            smallSDCursor.offsets.dv += smallSDConstInfo.dkvGroupStride;
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
        smallSDCursor.qPrefix += smallSDConstInfo.s1;
        smallSDCursor.kvPrefix += smallSDConstInfo.s2;
        smallSDCursor.s1s2Prefix += smallSDConstInfo.s1 * smallSDConstInfo.s2;
        smallSDCursor.s1s2AlignPrefix += smallSDConstInfo.s1 * smallSDConstInfo.s2Align16;
        smallSDCursor.offsets.q += smallSDConstInfo.qSStride;
        smallSDCursor.offsets.k += smallSDConstInfo.kvSStride;
        smallSDCursor.offsets.v += smallSDConstInfo.kvSStride;
        smallSDCursor.offsets.dy += smallSDConstInfo.qSStride;
        smallSDCursor.offsets.attention += smallSDConstInfo.qSStride;
        smallSDCursor.offsets.dq += smallSDConstInfo.qSStride;
        smallSDCursor.offsets.dk += smallSDConstInfo.kvSStride;
        smallSDCursor.offsets.dv += smallSDConstInfo.kvSStride;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::SetSmallSDPreloadArgs(
    const SmallSDRunInfo &runInfo, const SmallSDRunInfo &nextRunInfo, bool hasNext)
{
    this->preloadArgs.copyNext = hasNext;
    this->preloadArgs.copyCurrent = runInfo.taskId == 0;
    if (!hasNext) {
        return;
    }
    this->preloadArgs.nextQueryOffset = nextRunInfo.offsets.q;
    this->preloadArgs.nextDyOffset = nextRunInfo.offsets.dy;
    this->preloadArgs.nextMOrN = nextRunInfo.shape.s1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessVec1SmallSD(
    const SmallSDRunInfo &runInfo)
{
    this->vecBlock.ProcessVec1SmallSD(smallSDConstInfo, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::IssueMm1Mm2SmallSD(
    const SmallSDRunInfo &runInfo, const SmallSDRunInfo &nextRunInfo, bool hasNext, bool &needSyncDkMM)
{
    SetSmallSDPreloadArgs(runInfo, nextRunInfo, hasNext);

    if constexpr (KernelBaseClass::IS_DK_WRITE_UB) {
        if ASCEND_IS_AIC {
            if (needSyncDkMM) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
            }
        }
    }

    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmQKSmallSD(mm2ResTensor, smallSDConstInfo, runInfo, this->preloadArgs);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[runInfo.taskId & 1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[runInfo.taskId & 1]);
    }

    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmDyVSmallSD(mm1ResTensor, smallSDConstInfo, runInfo, this->preloadArgs);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[runInfo.taskId & 1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[runInfo.taskId & 1]);
    }

    this->vecBlock.CopyMaxSumSmallSD(smallSDConstInfo, runInfo, runInfo.taskId);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ComputeDqkvSmallSD(
    const SmallSDRunInfo &runInfo, bool &needSyncDkMM, int64_t taskId)
{
    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    this->vecBlock.ProcessVec2SmallSD(mm2ResTensor, smallSDConstInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
        }
    }
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3SmallSD(dSL1Buffer, mm1ResTensor, mm2ResTensor, smallSDConstInfo, runInfo);
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
        }
    }
    this->vecBlock.ProcessVec4SmallSD(pL1Buffer, mm2ResTensor, smallSDConstInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG);
    }

    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }

    if constexpr (KernelBaseClass::IS_DQ_WRITE_UB) {
        mm1ResTensor = this->mm1ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
        this->cubeBlock.template IterateMmDsKSmallSD<CALC_TYPE, KernelBaseClass::IS_DQ_WRITE_UB>(
            mm1ResTensor, dSL1Buffer, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
        }
        this->vecBlock.template ProcessMulsAndCastSmallSD<CALC_TYPE, KernelBaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
            mm1ResTensor, smallSDConstInfo, runInfo);
    } else {
        this->cubeBlock.template IterateMmDsKSmallSD<CALC_TYPE, KernelBaseClass::IS_DQ_WRITE_UB>(
            this->dqWorkSpaceGm, dSL1Buffer, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
        }
        this->vecBlock.template ProcessMulsAndCastSmallSD<CALC_TYPE, KernelBaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
            this->dqWorkSpaceGm, smallSDConstInfo, runInfo);
    }

    if constexpr (KernelBaseClass::IS_DK_WRITE_UB) {
        mm2ResTensor = this->mm2ResBuf[runInfo.taskId & 1].template Get<CALC_TYPE>();
        this->cubeBlock.template IterateMmDsQSmallSD<CALC_TYPE, KernelBaseClass::IS_DK_WRITE_UB>(
            mm2ResTensor, dSL1Buffer, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
        }
        this->vecBlock.template ProcessMulsAndCastSmallSD<CALC_TYPE, KernelBaseClass::IS_DK_WRITE_UB, DK_IDX>(
            mm2ResTensor, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
        }
    } else {
        this->cubeBlock.template IterateMmDsQSmallSD<CALC_TYPE, KernelBaseClass::IS_DK_WRITE_UB>(
            this->dkWorkSpaceGm, dSL1Buffer, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
        }
        this->vecBlock.template ProcessMulsAndCastSmallSD<CALC_TYPE, KernelBaseClass::IS_DK_WRITE_UB, DK_IDX>(
            this->dkWorkSpaceGm, smallSDConstInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
        }
    }

    this->cubeBlock.template IterateMmPDySmallSD<OUTDTYPE, KernelBaseClass::IS_DV_WRITE_UB>(
        this->dvGm, pL1Buffer, smallSDConstInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
    }
    needSyncDkMM = true;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessSingleGroupSmallSD()
{
    SmallSDRunInfo runInfo;
    SmallSDRunInfo nextRunInfo = {};
    bool needSyncDkMM = false;

    this->isLastLoop = false;
    PrepareSmallSDRunInfo(runInfo, 0);
    IssueMm1Mm2SmallSD(runInfo, nextRunInfo, false, needSyncDkMM);

    this->isLastLoop = true;
    ProcessVec1SmallSD(runInfo);
    ComputeDqkvSmallSD(runInfo, needSyncDkMM, 1);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::ProcessMultiGroupSmallSD(int64_t groupCount)
{
    SmallSDRunInfo runInfos[2];
    SmallSDRunInfo nextRunInfo = {};
    bool needSyncDkMM = false;
    for (int64_t taskId = 0; taskId <= groupCount; ++taskId) {
        this->isLastLoop = taskId == groupCount;
        if (taskId > 0) {
            ProcessVec1SmallSD(runInfos[(taskId + 1) & 1]); // v1: softmaxGrad
        }
        if (!this->isLastLoop) {
            PrepareSmallSDRunInfo(runInfos[taskId & 1], taskId);
            const bool hasNext = (taskId + 1 < groupCount);
            if (hasNext) {
                AdvanceSmallSDCursor();
                PrepareSmallSDRunInfo(nextRunInfo, taskId + 1);
            }
            IssueMm1Mm2SmallSD(runInfos[taskId & 1], nextRunInfo, hasNext, needSyncDkMM);
        }
        if (taskId > 0) {
            ComputeDqkvSmallSD(runInfos[(taskId + 1) & 1], needSyncDkMM, taskId);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelSmallSD<CubeBlockType, VecBlockType>::Process()
{
    static_assert(SPLIT_AXIS == BN2, "SmallSD only supports BN2 split axis.");
    static_assert(!IS_BN2_MULTIBLK, "SmallSD does not support BN2 multi block.");
    static_assert(!IS_ATTEN_MASK && !IS_PSE && !IS_DROP, "SmallSD does not support optional inputs.");
    static_assert(!IS_ROPE && !IS_D_NO_EQUAL, "SmallSD requires D == Dv without rope.");
    static_assert(!IS_NZ_OUT && !IS_TND_SWIZZLE, "SmallSD does not support NZ output or TND swizzle.");
    static_assert(DETER_SPARSE_TYPE == NO_DETER, "SmallSD does not support deterministic sparse mode.");
    static_assert((IsSameType<INPUT_TYPE, half>::value && IsSameType<OUTDTYPE, half>::value) ||
                      (IsSameType<INPUT_TYPE, bfloat16_t>::value && IsSameType<OUTDTYPE, bfloat16_t>::value),
                  "SmallSD only supports FP16/BF16 with matching output dtype.");
    static_assert(s1TemplateType == S1TemplateType::Aligned128 && s2TemplateType == S2TemplateType::Aligned128,
                  "SmallSD only supports 128x128 S template.");
    static_assert(dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128,
                  "SmallSD only supports D template 64 or 128.");

    const int64_t groupCount = smallSDConstInfo.groupCount;
    if (groupCount == 0) {
        return;
    }
    InitSmallSDCursor();

    if (groupCount == 1) {
        ProcessSingleGroupSmallSD();
        return;
    }
    ProcessMultiGroupSmallSD(groupCount);
}

} // namespace FagBaseApi
#endif
