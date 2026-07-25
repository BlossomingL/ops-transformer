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
    constexpr static uint32_t CUBE_BASEM = BaseClass::CUBE_BASEM;
    constexpr static uint32_t CUBE_BASEN = BaseClass::CUBE_BASEN;
    constexpr static uint32_t HEAD_DIM_ALIGN = BaseClass::HEAD_DIM_ALIGN;
    constexpr static uint32_t DQ_L0_SPLIT_K = BaseClass::DQ_L0_SPLIT_K;
    constexpr static uint32_t DKV_L0_SPLIT_K = BaseClass::DKV_L0_SPLIT_K;
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
    baseBlock.InitCubeBuffer(compatConstInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmQKSmallSD(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo,
    PreloadArgs<IS_ROPE> &preloadArgs)
{
    MutexBuffer<BufferType::L1> qL1Buffer;
    MutexBuffer<BufferType::L1> qL1NextBuffer;
    MutexBuffer<BufferType::L1> kL1Buffer;
    Nd2NzParams nd2NzParams;

    if constexpr (BaseClass::IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            qL1Buffer = baseBlock.qL1Buf.Get();
        } else {
            qL1Buffer = baseBlock.qL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            qL1NextBuffer = baseBlock.qL1Buf.Get();
        }
    } else if constexpr (BaseClass::IS_L1_REUSE) {
        qL1Buffer = baseBlock.qL1Buf.Get();
    } else {
        qL1Buffer = baseBlock.commonL1Buf.Get();
    }

    if (!BaseClass::IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        qL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.shape.s1;
        nd2NzParams.dValue = smallSDConstInfo.d;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.shape.s1);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(qL1Tensor, baseBlock.queryGm[runInfo.offsets.q], nd2NzParams);
        qL1Buffer.UnlockProd();
    }

    qL1Buffer.LockCons();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = runInfo.shape.s2;
    uint32_t gmNOffset = 0;
    uint32_t ubOffset = 0;

    bool isCopyRight = true;
    if constexpr (BaseClass::IS_L1_REUSE || BaseClass::IS_L1_PRELOAD) {
        kL1Buffer = baseBlock.kL1Buf.Get();
    } else {
        kL1Buffer = baseBlock.commonL1Buf.Get();
    }

    if (isCopyRight) {
        kL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = realN;
        nd2NzParams.dValue = smallSDConstInfo.d;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.kvRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(realN);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(kL1Tensor, baseBlock.keyGm[runInfo.offsets.k + gmNOffset], nd2NzParams);
        kL1Buffer.UnlockProd();
    }

    if constexpr (BaseClass::IS_L1_PRELOAD) {
        if (preloadArgs.copyNext) {
            qL1NextBuffer.LockProd();
            LocalTensor<INPUT_TYPE> qL1Tensor = qL1NextBuffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = preloadArgs.nextMOrN;
            nd2NzParams.dValue = smallSDConstInfo.d;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
            nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, baseBlock.queryGm[preloadArgs.nextQueryOffset], nd2NzParams);
            qL1NextBuffer.UnlockProd();
        }
    }

    MutexBuffer<BufferType::L0C> mm2L0CBuffer;
    if constexpr (BaseClass::IS_DKV_RESIDENT_L0C) {
        mm2L0CBuffer = baseBlock.mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm2L0CBuffer = baseBlock.commonl0CBuf.Get();
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm2L0CBuffer.LockProd();
    }
    MMParam param = {
        static_cast<uint32_t>(runInfo.shape.s1),
        realN,
        static_cast<uint32_t>(smallSDConstInfo.d),
        false,
        true,
        true,
        true,
        BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };

    kL1Buffer.LockCons();
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
        BaseClass::L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
        qL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), baseBlock.l0aBuf,
        baseBlock.l0bBuf, mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
    kL1Buffer.UnlockCons();

    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm2L0CBuffer.UnlockProd();
        mm2L0CBuffer.LockCons();
    }

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = realN;
    fixpipeParams.mSize = (runInfo.shape.s1 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = CUBE_BASEN;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor[ubOffset],
                                                        mm2L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm2L0CBuffer.UnlockCons();
    }
    qL1Buffer.UnlockCons();
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDyVSmallSD(
    LocalTensor<CALC_TYPE> &mm1ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo,
    PreloadArgs<IS_ROPE> &preloadArgs)
{
    MutexBuffer<BufferType::L1> dyL1Buffer;
    MutexBuffer<BufferType::L1> dyL1NextBuffer;
    MutexBuffer<BufferType::L1> vL1Buffer;
    Nd2NzParams nd2NzParams;

    if constexpr (BaseClass::IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            dyL1Buffer = baseBlock.dYL1Buf.Get();
        } else {
            dyL1Buffer = baseBlock.dYL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            dyL1NextBuffer = baseBlock.dYL1Buf.Get();
        }
    } else if constexpr (BaseClass::IS_L1_REUSE) {
        dyL1Buffer = baseBlock.dYL1Buf.Get();
    } else {
        dyL1Buffer = baseBlock.commonL1Buf.Get();
    }

    if (!BaseClass::IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        dyL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.shape.s1;
        nd2NzParams.dValue = smallSDConstInfo.dv;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.shape.s1);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dyL1Tensor, baseBlock.dyGm[runInfo.offsets.dy], nd2NzParams);
        dyL1Buffer.UnlockProd();
    }

    dyL1Buffer.LockCons();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = runInfo.shape.s2;
    uint32_t gmNOffset = 0;
    uint32_t ubOffset = 0;

    bool isCopyRight = true;
    if constexpr (BaseClass::IS_L1_REUSE || BaseClass::IS_L1_PRELOAD) {
        vL1Buffer = baseBlock.vL1Buf.Get();
    } else {
        vL1Buffer = baseBlock.commonL1Buf.Get();
    }

    if (isCopyRight) {
        vL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = realN;
        nd2NzParams.dValue = smallSDConstInfo.dv;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.kvRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(realN);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(vL1Tensor, baseBlock.valueGm[runInfo.offsets.v + gmNOffset], nd2NzParams);
        vL1Buffer.UnlockProd();
    }

    if constexpr (BaseClass::IS_L1_PRELOAD) {
        if (preloadArgs.copyNext) {
            dyL1NextBuffer.LockProd();
            LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1NextBuffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = preloadArgs.nextMOrN;
            nd2NzParams.dValue = smallSDConstInfo.dv;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
            nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(dyL1Tensor, baseBlock.dyGm[preloadArgs.nextDyOffset], nd2NzParams);
            dyL1NextBuffer.UnlockProd();
        }
    }

    MutexBuffer<BufferType::L0C> mm1L0CBuffer;
    if constexpr (BaseClass::IS_DKV_RESIDENT_L0C) {
        mm1L0CBuffer = baseBlock.mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm1L0CBuffer = baseBlock.commonl0CBuf.Get();
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm1L0CBuffer.LockProd();
    }
    MMParam param = {
        static_cast<uint32_t>(runInfo.shape.s1),
        realN,
        static_cast<uint32_t>(smallSDConstInfo.dv),
        false,
        true,
        true,
        true,
        BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };

    vL1Buffer.LockCons();
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
        BaseClass::L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
        dyL1Buffer.template GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), baseBlock.l0aBuf,
        baseBlock.l0bBuf, mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
    vL1Buffer.UnlockCons();

    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm1L0CBuffer.UnlockProd();
        mm1L0CBuffer.LockCons();
    }

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = realN;
    fixpipeParams.mSize = (runInfo.shape.s1 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = CUBE_BASEN;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.unitFlag = BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor[ubOffset],
                                                        mm1L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm1L0CBuffer.UnlockCons();
    }
    dyL1Buffer.UnlockCons();
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDsKSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    uint32_t realN = static_cast<uint32_t>(smallSDConstInfo.d);
    constexpr uint64_t gmNOffset = 0;
    MutexBuffer<BufferType::L1> kL1Buffer;
    LocalTensor<INPUT_TYPE> kL1Tensor;
    if constexpr (BaseClass::IS_L1_PRELOAD || BaseClass::IS_L1_REUSE) {
        kL1Buffer = baseBlock.kL1Buf.GetReused(false);
        kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        kL1Buffer = baseBlock.commonL1Buf.Get();
        kL1Buffer.LockProd();
        kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.shape.s2;
        nd2NzParams.dValue = smallSDConstInfo.d;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.kvRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.shape.s2);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(kL1Tensor, baseBlock.keyGm[runInfo.offsets.k], nd2NzParams);
        kL1Buffer.UnlockProd();
    }
    kL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> mm3L0CBuffer;
    if constexpr (BaseClass::IS_DKV_RESIDENT_L0C) {
        mm3L0CBuffer = baseBlock.mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm3L0CBuffer = baseBlock.commonl0CBuf.Get();
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm3L0CBuffer.LockProd();
    }
    MMParam param = {
        static_cast<uint32_t>(runInfo.shape.s1),
        realN,
        static_cast<uint32_t>(runInfo.shape.s2),
        false,
        false,
        true,
        true,
        BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, baseBlock.l0aBuf, baseBlock.l0bBuf,
        mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
    kL1Buffer.UnlockCons();

    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm3L0CBuffer.UnlockProd();
        mm3L0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (realN + 7) >> 3 << 3;
    fixpipeParams.mSize = (runInfo.shape.s1 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.dAlign16;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
    if constexpr (IS_WRITE_UB) {
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[gmNOffset], mm3L0CBuffer.GetTensor<CALC_TYPE>(),
                                                fixpipeParams);
    } else {
        fixpipeParams.nSize = smallSDConstInfo.d;
        fixpipeParams.mSize = runInfo.shape.s1;
        fixpipeParams.dualDstCtl = 0;
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(
            outTensor[GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN + gmNOffset],
            mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        mm3L0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmDsQSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    uint32_t realN = static_cast<uint32_t>(smallSDConstInfo.d);
    constexpr uint64_t gmNOffset = 0;
    LocalTensor<INPUT_TYPE> dsL1Tensor = dSL1Buffer.GetTensor<INPUT_TYPE>();
    MutexBuffer<BufferType::L1> qL1Buffer;
    LocalTensor<INPUT_TYPE> qL1Tensor;
    if constexpr (BaseClass::IS_L1_PRELOAD || BaseClass::IS_L1_REUSE) {
        qL1Buffer = baseBlock.qL1Buf.GetReused();
        qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        qL1Buffer = baseBlock.commonL1Buf.Get();
        qL1Buffer.LockProd();
        qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.shape.s1;
        nd2NzParams.dValue = smallSDConstInfo.d;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.shape.s1);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(qL1Tensor, baseBlock.queryGm[runInfo.offsets.q], nd2NzParams);
        qL1Buffer.UnlockProd();
    }
    qL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> dkL0CBuffer;
    if constexpr (BaseClass::IS_DKV_RESIDENT_L0C) {
        dkL0CBuffer = baseBlock.dkL0CBuf.Get();
    } else {
        dkL0CBuffer = baseBlock.commonl0CBuf.Get();
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dkL0CBuffer.LockProd();
    }
    MMParam param = {
        static_cast<uint32_t>(runInfo.shape.s2),
        realN,
        static_cast<uint32_t>(runInfo.shape.s1),
        true,
        false,
        true,
        true,
        BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        dsL1Tensor, qL1Tensor, baseBlock.l0aBuf, baseBlock.l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(), param);
    qL1Buffer.UnlockCons();

    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dkL0CBuffer.UnlockProd();
        dkL0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (realN + 7) >> 3 << 3;
    fixpipeParams.mSize = (runInfo.shape.s2 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.dAlign16;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
    if constexpr (IS_WRITE_UB) {
        Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[gmNOffset], dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                                fixpipeParams);
    } else {
        fixpipeParams.nSize = smallSDConstInfo.d;
        fixpipeParams.mSize = runInfo.shape.s2;
        fixpipeParams.dualDstCtl = 0;
        Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(
            outTensor[GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN + gmNOffset],
            dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dkL0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::IterateMmPDySmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    uint32_t realN = static_cast<uint32_t>(smallSDConstInfo.dv);
    constexpr uint64_t gmNOffset = 0;
    MutexBuffer<BufferType::L1> dYL1Buffer;
    LocalTensor<OUTDTYPE> dYL1Tensor;
    if constexpr (BaseClass::IS_L1_PRELOAD || BaseClass::IS_L1_REUSE) {
        dYL1Buffer = baseBlock.dYL1Buf.GetReused();
        dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        dYL1Buffer = baseBlock.commonL1Buf.Get();
        dYL1Buffer.LockProd();
        dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.shape.s1;
        nd2NzParams.dValue = smallSDConstInfo.dv;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = smallSDConstInfo.qRowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.shape.s1);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dYL1Tensor, baseBlock.dyGm[runInfo.offsets.dy + gmNOffset], nd2NzParams);
        dYL1Buffer.UnlockProd();
    }
    dYL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> dvL0CBuffer;
    if constexpr (BaseClass::IS_DKV_RESIDENT_L0C) {
        dvL0CBuffer = baseBlock.dvL0CBuf.Get();
    } else {
        dvL0CBuffer = baseBlock.commonl0CBuf.Get();
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dvL0CBuffer.LockProd();
    }
    MMParam param = {
        static_cast<uint32_t>(runInfo.shape.s2),
        realN,
        static_cast<uint32_t>(runInfo.shape.s1),
        true,
        false,
        true,
        true,
        BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, baseBlock.l0aBuf, baseBlock.l0bBuf,
        dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
    dYL1Buffer.UnlockCons();

    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dvL0CBuffer.UnlockProd();
        dvL0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = runInfo.shape.s2;
    fixpipeParams.nSize = realN;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.kvRowStride;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = BaseClass::ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
    if constexpr (!IS_WRITE_UB) {
        if constexpr (IsSameType<T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        }
        Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.offsets.dv + gmNOffset],
                                                dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    } else {
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[gmNOffset],
                                                dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    }
    if constexpr (!BaseClass::ENABLE_UNITFLAG) {
        dvL0CBuffer.UnlockCons();
    }
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
