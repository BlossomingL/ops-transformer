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
 * \file flash_attention_score_grad_block_cube_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H

#include "flash_attention_score_grad_common_small_sd.h"
#include "flash_attention_score_grad_block_cube.h"

namespace FagBaseApi {

TEMPLATES_DEF
class FAGBlockCubeSmallSD {
public:
    using BaseClass = FAGBlockCube<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockCubeSmallSD(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              MutexBufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffer(FagConstInfo &compatConstInfo);
    __aicore__ inline void IterateMmQKSmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs);
    __aicore__ inline void IterateMmDyVSmallSD(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDySmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo);
private:
    __aicore__ inline void BuildCompatRunInfo(FagRunInfo &compatRunInfo, const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo);
    FagConstInfo compatConstInfo;
    BaseClass baseBlock;
};

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::SetCubeBlockParams(
    TPipe *pipe, FagTilingType tilingData, MutexBufferManager<BufferType::L1> *l1BuffMgr)
{
    baseBlock.SetCubeBlockParams(pipe, tilingData, l1BuffMgr);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR dv, GM_ADDR workspace)
{
    baseBlock.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &compatConstInfo)
{
    this->compatConstInfo = compatConstInfo;
    baseBlock.InitCubeBuffer(this->compatConstInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::BuildCompatRunInfo(
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
    compatRunInfo.commonRunInfo.vecCoreOffset = 0;
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
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmQKSmallSD(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo,
    PreloadArgs<IS_ROPE> &preloadArgs)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.IterateMmQK(mm2ResTensor, compatConstInfo, compatRunInfo, preloadArgs);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDyVSmallSD(
    LocalTensor<CALC_TYPE> &mm1ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo,
    PreloadArgs<IS_ROPE> &preloadArgs)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.IterateMmDyV(mm1ResTensor, compatConstInfo, compatRunInfo, preloadArgs);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDsKSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.template IterateMmDsK<T, IS_WRITE_UB>(outTensor, dSL1Buffer, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDsQSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.template IterateMmDsQ<T, IS_WRITE_UB>(outTensor, dSL1Buffer, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmPDySmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    FagRunInfo compatRunInfo;
    BuildCompatRunInfo(compatRunInfo, smallSDConstInfo, runInfo);
    baseBlock.template IterateMmPDy<T, IS_WRITE_UB>(outTensor, pL1Buffer, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
class FAGBlockCubeSmallSDDummy {
public:
    __aicore__ inline FAGBlockCubeSmallSDDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              MutexBufferManager<BufferType::L1> *l1BuffMgr){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace){};
    __aicore__ inline void InitCubeBuffer(FagConstInfo &compatConstInfo){};
    __aicore__ inline void IterateMmQKSmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             const SmallSDRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs){};
    __aicore__ inline void IterateMmDyVSmallSD(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs){};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo){};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo){};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDySmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
                                              const SmallSDConstInfo &smallSDConstInfo,
                                              const SmallSDRunInfo &runInfo){};
};

DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeSmallSD);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeSmallSDDummy);

} // namespace FagBaseApi
#endif
