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
 * \file flash_attention_score_grad_block_vec_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H

#include "flash_attention_score_grad_common_small_sd.h"
#include "flash_attention_score_grad_block_vec.h"

namespace FagBaseApi {

TEMPLATES_DEF
class FAGBlockVecSmallSD {
public:
    using BaseClass = FAGBlockVec<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockVecSmallSD(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo,
                                             DropMaskInfo &dropInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace);
    __aicore__ inline void SetOldDeterFp32Param(FagConstInfo &compatConstInfo);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void SetSmallSDCompatConstInfo(FagConstInfo &compatConstInfo);
    __aicore__ inline void ProcessVec1SmallSD(const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo);
    __aicore__ inline void CopyMaxSumSmallSD(const SmallSDConstInfo &smallSDConstInfo,
                                            const SmallSDRunInfo &runInfo, int64_t taskId);
    __aicore__ inline void ProcessVec2SmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo);
    __aicore__ inline void ProcessVec3SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm1ResTensor,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo);
    __aicore__ inline void ProcessVec4SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCastSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                                    const SmallSDConstInfo &smallSDConstInfo,
                                                    const SmallSDRunInfo &runInfo);
private:
    __aicore__ inline void BuildCompatRunInfo(FagRunInfo &compatRunInfo, const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo);
    FagConstInfo compatConstInfo;
    uint32_t vSubBlockIdxForCompat = 0;
    BaseClass baseBlock;
};

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::SetVecBlockParams(
    TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx, uint32_t cBlockIdx, uint32_t vSubBlockIdx,
    AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo, DropMaskInfo &dropInfo)
{
    vSubBlockIdxForCompat = vSubBlockIdx;
    baseBlock.SetVecBlockParams(pipe, tilingData, vBlockIdx, cBlockIdx, vSubBlockIdx, attenMaskInfo, pseInfo,
                                dropInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR softmaxMax,
    GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace)
{
    baseBlock.InitGlobalBuffer(value, dy, y, pseShift, dropMask, attenMask, softmaxMax, softmaxSum, deqScaleQ,
                               deqScaleK, deqScaleV, deqScaleDy, dq, dk, dv, dqRope, dkRope, sink, dsink, workspace);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::SetOldDeterFp32Param(FagConstInfo &compatConstInfo)
{
    baseBlock.SetOldDeterFp32Param(compatConstInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::InitUbBuffer()
{
    baseBlock.InitUbBuffer();
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::SetSmallSDCompatConstInfo(FagConstInfo &compatConstInfo)
{
    this->compatConstInfo = compatConstInfo;
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::BuildCompatRunInfo(
    FagRunInfo &compatRunInfo, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    compatRunInfo = {};
    compatRunInfo.commonRunInfo.boIdx = runInfo.batchIdx;
    compatRunInfo.commonRunInfo.n2oIdx = runInfo.n2Idx;
    compatRunInfo.commonRunInfo.goIdx = 0;
    compatRunInfo.commonRunInfo.s1oIdx = 0;
    compatRunInfo.commonRunInfo.taskId = runInfo.taskId;
    compatRunInfo.commonRunInfo.taskIdMod2 = runInfo.taskId & 1;
    compatRunInfo.commonRunInfo.s1RealSize = runInfo.shape.s1;
    compatRunInfo.commonRunInfo.s2RealSize = runInfo.shape.s2;
    compatRunInfo.commonRunInfo.actualS1Size = runInfo.shape.s1;
    compatRunInfo.commonRunInfo.actualS2Size = runInfo.shape.s2;
    compatRunInfo.commonRunInfo.halfS1RealSize = runInfo.shape.halfS1;
    compatRunInfo.commonRunInfo.firstHalfS1RealSize = runInfo.shape.firstHalfS1;
    compatRunInfo.commonRunInfo.s2SizeAcc = runInfo.kvPrefix;
    compatRunInfo.commonRunInfo.b1SSOffsetAlign = runInfo.s1s2AlignPrefix;
    compatRunInfo.commonRunInfo.b1SSOffset = runInfo.s1s2Prefix;
    compatRunInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.s1s2Prefix;
    compatRunInfo.commonRunInfo.s2StartIdx = 0;
    compatRunInfo.commonRunInfo.s2AlignedSize = runInfo.shape.s2Align16;
    compatRunInfo.commonRunInfo.vecCoreOffset = vSubBlockIdxForCompat * runInfo.shape.firstHalfS1;
    compatRunInfo.commonRunInfo.queryOffset = runInfo.offsets.q;
    compatRunInfo.commonRunInfo.keyOffset = runInfo.offsets.k;
    compatRunInfo.commonRunInfo.valueOffset = runInfo.offsets.v;
    compatRunInfo.s2oIdx = 0;
    compatRunInfo.s2CvBegin = 0;
    compatRunInfo.s2CvEnd = runInfo.shape.s2;
    compatRunInfo.halfS2RealSize = runInfo.shape.halfS2;
    compatRunInfo.firstHalfS2RealSize = runInfo.shape.firstHalfS2;
    compatRunInfo.dyOffset = runInfo.offsets.dy;
    compatRunInfo.queryOffsetWithRope = runInfo.offsets.q;
    compatRunInfo.keyOffsetWithRope = runInfo.offsets.k;
    compatRunInfo.queryOffsetWithRopeForMm12 = runInfo.offsets.q;
    compatRunInfo.keyOffsetWithRopeForMm12 = runInfo.offsets.k;
    compatRunInfo.maxsumOffset = runInfo.offsets.softmaxMax;
    compatRunInfo.lastBatchIdx = runInfo.batchIdx;
    compatRunInfo.lastBatchTotalS1BOffset = runInfo.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d;
    compatRunInfo.lastBatchTotalS2BOffset = runInfo.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.d;
    compatRunInfo.lastBatchTotalS1BOffsetForDv = runInfo.qPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.dv;
    compatRunInfo.lastBatchTotalS2BOffsetForDv = runInfo.kvPrefix * smallSDConstInfo.n2Size * smallSDConstInfo.dv;
    compatRunInfo.lastBatchTotalS1S2SizeAlign = runInfo.s1s2AlignPrefix;
    compatRunInfo.lastBatchTotalS1S2Size = runInfo.s1s2Prefix;
    compatRunInfo.lastBatchTotalS2Size = runInfo.kvPrefix;
    compatRunInfo.isNextS2IdxNoChange = false;
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec1SmallSD(
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.ProcessVec1(compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::CopyMaxSumSmallSD(
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo, int64_t taskId)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.CopyMaxSum(compatConstInfo, compatRunInfo, taskId);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec2SmallSD(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.ProcessVec2(mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec3SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.ProcessVec3(dstBuffer, mm1ResTensor, mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec4SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.ProcessVec4(dstBuffer, mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessMulsAndCastSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.template ProcessMulsAndCast<T, IS_WRITE_UB, MM_IDX>(inputTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
class FAGBlockVecSmallSDDummy {
public:
    __aicore__ inline FAGBlockVecSmallSDDummy(){};
    __aicore__ inline void InitUbBuffer(){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo,
                                             DropMaskInfo &dropInfo){};
    __aicore__ inline void SetOldDeterFp32Param(FagConstInfo &compatConstInfo){};
    __aicore__ inline void SetSmallSDCompatConstInfo(FagConstInfo &compatConstInfo){};
    __aicore__ inline void ProcessVec1SmallSD(const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo){};
    __aicore__ inline void CopyMaxSumSmallSD(const SmallSDConstInfo &smallSDConstInfo,
                                            const SmallSDRunInfo &runInfo, int64_t taskId){};
    __aicore__ inline void ProcessVec2SmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo){};
    __aicore__ inline void ProcessVec3SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm1ResTensor,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo){};
    __aicore__ inline void ProcessVec4SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo){};
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCastSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                                    const SmallSDConstInfo &smallSDConstInfo,
                                                    const SmallSDRunInfo &runInfo){};
};

} // namespace FagBaseApi
#endif
