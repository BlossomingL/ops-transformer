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
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    constexpr static bool IS_L1_REUSE =
        GET_IS_L1_REUSE<INPUT_TYPE>(HEAD_DIM_ALIGN, IS_DETER_OLD(DETER_SPARSE_TYPE), false);
    constexpr static bool IS_L1_PRELOAD = GET_IS_L1_PRELOAD<INPUT_TYPE>(
        HEAD_DIM_ALIGN, SPLIT_AXIS, IS_DETER_OLD(DETER_SPARSE_TYPE), IS_TND, false, IS_ROPE);
    constexpr static bool IS_DKV_RESIDENT_L0C =
        IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) && !IS_FP32_INPUT;
    constexpr static uint32_t DQ_L0_SPLIT_K = GET_DQ_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEM, HEAD_DIM_ALIGN>();
    constexpr static uint32_t DKV_L0_SPLIT_K = GET_DKV_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEN, HEAD_DIM_ALIGN>();
    constexpr static bool ENABLE_UNITFLAG =
        HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned768) && !IS_DETER_OLD(DETER_SPARSE_TYPE);
    __aicore__ inline FAGBlockCubeSmallSD(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, MutexBufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    template <typename InitInfo>
    __aicore__ inline void InitCubeBuffer(InitInfo &);
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
    GlobalTensor<INPUT_TYPE> queryGm, keyGm, valueGm, queryRopeGm, keyRopeGm;
    GlobalTensor<OUTDTYPE> dyGm;
    TPipe *pipe;
    MutexBufferManager<BufferType::L1> *l1BufferManagerPtr;
    typename std::conditional<IS_L1_REUSE,
                              typename DyL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, false>::TYPE,
                              std::nullptr_t>::type dYL1Buf;
    typename std::conditional<IS_L1_REUSE, MutexBuffersPolicySingleBuffer<BufferType::L1>, std::nullptr_t>::type vL1Buf;
    typename std::conditional<IS_L1_REUSE,
                              typename QL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, false>::TYPE,
                              std::nullptr_t>::type qL1Buf;
    typename std::conditional<IS_L1_REUSE,
                              typename KL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, false>::TYPE,
                              std::nullptr_t>::type kL1Buf;
    typename std::conditional<!IS_L1_REUSE, MutexBuffersPolicyDB<BufferType::L1>, std::nullptr_t>::type commonL1Buf;
    MutexBufferManager<BufferType::L0A> l0aBufferManager;
    MutexBufferManager<BufferType::L0B> l0bBufferManager;
    MutexBuffersPolicyDB<BufferType::L0A> l0aBuf;
    MutexBuffersPolicyDB<BufferType::L0B> l0bBuf;
    MutexBufferManager<BufferType::L0C> l0cBufferManager;
    using L0CType = typename mm1Mm2Mm3L0CBuffSelector<HEAD_DIM_ALIGN>::TYPE;
    typename std::conditional<IS_DKV_RESIDENT_L0C, L0CType, std::nullptr_t>::type mm1Mm2Mm3L0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, MutexBuffersPolicySingleBuffer<BufferType::L0C>,
                              std::nullptr_t>::type dkL0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, MutexBuffersPolicySingleBuffer<BufferType::L0C>,
                              std::nullptr_t>::type dvL0CBuf;
    typename std::conditional<!IS_DKV_RESIDENT_L0C, typename CommonL0CBufSelector<HEAD_DIM_ALIGN>::TYPE,
                              std::nullptr_t>::type commonl0CBuf;
};

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::SetCubeBlockParams(
    TPipe *pipe, MutexBufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR dv, GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    queryRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
}

TEMPLATES_DEF
template <typename InitInfo>
__aicore__ inline void FAGBlockCubeSmallSD<TEMPLATE_ARGS>::InitCubeBuffer(InitInfo &)
{
    if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
        dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        vL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        qL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        kL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    } else {
        commonL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    }

    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm1Mm2Mm3L0CBuf.Init(l0cBufferManager, CUBE_BASEN > HEAD_DIM_ALIGN ?
                                                   CUBE_BASEM * CUBE_BASEN * sizeof(float) :
                                                   CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(float));
        dkL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
        dvL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
    } else {
        commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE / NUM_TWO);
    }
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

    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            qL1Buffer = qL1Buf.Get();
        } else {
            qL1Buffer = qL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            qL1NextBuffer = qL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        qL1Buffer = qL1Buf.Get();
    } else {
        qL1Buffer = commonL1Buf.Get();
    }

    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
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
        DataCopy(qL1Tensor, queryGm[runInfo.offsets.q], nd2NzParams);
        qL1Buffer.UnlockProd();
    }

    qL1Buffer.LockCons();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = runInfo.shape.s2;
    uint32_t gmNOffset = 0;
    uint32_t ubOffset = 0;

    bool isCopyRight = true;
    if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
        kL1Buffer = kL1Buf.Get();
    } else {
        kL1Buffer = commonL1Buf.Get();
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
        DataCopy(kL1Tensor, keyGm[runInfo.offsets.k + gmNOffset], nd2NzParams);
        kL1Buffer.UnlockProd();
    }

    if constexpr (IS_L1_PRELOAD) {
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
            DataCopy(qL1Tensor, queryGm[preloadArgs.nextQueryOffset], nd2NzParams);
            qL1NextBuffer.UnlockProd();
        }
    }

    MutexBuffer<BufferType::L0C> mm2L0CBuffer;
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm2L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm2L0CBuffer = commonl0CBuf.Get();
    }
    if constexpr (!ENABLE_UNITFLAG) {
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
        ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };

    kL1Buffer.LockCons();
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
        L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
        qL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf,
        l0bBuf, mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
    kL1Buffer.UnlockCons();

    if constexpr (!ENABLE_UNITFLAG) {
        mm2L0CBuffer.UnlockProd();
        mm2L0CBuffer.LockCons();
    }

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = realN;
    fixpipeParams.mSize = (runInfo.shape.s1 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = CUBE_BASEN;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor[ubOffset],
                                                        mm2L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    if constexpr (!ENABLE_UNITFLAG) {
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

    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            dyL1Buffer = dYL1Buf.Get();
        } else {
            dyL1Buffer = dYL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            dyL1NextBuffer = dYL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        dyL1Buffer = dYL1Buf.Get();
    } else {
        dyL1Buffer = commonL1Buf.Get();
    }

    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
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
        DataCopy(dyL1Tensor, dyGm[runInfo.offsets.dy], nd2NzParams);
        dyL1Buffer.UnlockProd();
    }

    dyL1Buffer.LockCons();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = runInfo.shape.s2;
    uint32_t gmNOffset = 0;
    uint32_t ubOffset = 0;

    bool isCopyRight = true;
    if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
        vL1Buffer = vL1Buf.Get();
    } else {
        vL1Buffer = commonL1Buf.Get();
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
        DataCopy(vL1Tensor, valueGm[runInfo.offsets.v + gmNOffset], nd2NzParams);
        vL1Buffer.UnlockProd();
    }

    if constexpr (IS_L1_PRELOAD) {
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
            DataCopy(dyL1Tensor, dyGm[preloadArgs.nextDyOffset], nd2NzParams);
            dyL1NextBuffer.UnlockProd();
        }
    }

    MutexBuffer<BufferType::L0C> mm1L0CBuffer;
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm1L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm1L0CBuffer = commonl0CBuf.Get();
    }
    if constexpr (!ENABLE_UNITFLAG) {
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
        ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };

    vL1Buffer.LockCons();
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
        L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
        dyL1Buffer.template GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf,
        l0bBuf, mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
    vL1Buffer.UnlockCons();

    if constexpr (!ENABLE_UNITFLAG) {
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
    fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor[ubOffset],
                                                        mm1L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    if constexpr (!ENABLE_UNITFLAG) {
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
    if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
        kL1Buffer = kL1Buf.GetReused(false);
        kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        kL1Buffer = commonL1Buf.Get();
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
        DataCopy(kL1Tensor, keyGm[runInfo.offsets.k], nd2NzParams);
        kL1Buffer.UnlockProd();
    }
    kL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> mm3L0CBuffer;
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm3L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
    } else {
        mm3L0CBuffer = commonl0CBuf.Get();
    }
    if constexpr (!ENABLE_UNITFLAG) {
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
        ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
        mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
    kL1Buffer.UnlockCons();

    if constexpr (!ENABLE_UNITFLAG) {
        mm3L0CBuffer.UnlockProd();
        mm3L0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (realN + 7) >> 3 << 3;
    fixpipeParams.mSize = (runInfo.shape.s1 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.dAlign16;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
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
    if constexpr (!ENABLE_UNITFLAG) {
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
    if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
        qL1Buffer = qL1Buf.GetReused();
        qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        qL1Buffer = commonL1Buf.Get();
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
        DataCopy(qL1Tensor, queryGm[runInfo.offsets.q], nd2NzParams);
        qL1Buffer.UnlockProd();
    }
    qL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> dkL0CBuffer;
    if constexpr (IS_DKV_RESIDENT_L0C) {
        dkL0CBuffer = dkL0CBuf.Get();
    } else {
        dkL0CBuffer = commonl0CBuf.Get();
    }
    if constexpr (!ENABLE_UNITFLAG) {
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
        ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(), param);
    qL1Buffer.UnlockCons();

    if constexpr (!ENABLE_UNITFLAG) {
        dkL0CBuffer.UnlockProd();
        dkL0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (realN + 7) >> 3 << 3;
    fixpipeParams.mSize = (runInfo.shape.s2 + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.dAlign16;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
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
    if constexpr (!ENABLE_UNITFLAG) {
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
    if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
        dYL1Buffer = dYL1Buf.GetReused();
        dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
    } else {
        Nd2NzParams nd2NzParams;
        dYL1Buffer = commonL1Buf.Get();
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
        DataCopy(dYL1Tensor, dyGm[runInfo.offsets.dy + gmNOffset], nd2NzParams);
        dYL1Buffer.UnlockProd();
    }
    dYL1Buffer.LockCons();

    MutexBuffer<BufferType::L0C> dvL0CBuffer;
    if constexpr (IS_DKV_RESIDENT_L0C) {
        dvL0CBuffer = dvL0CBuf.Get();
    } else {
        dvL0CBuffer = commonl0CBuf.Get();
    }
    if constexpr (!ENABLE_UNITFLAG) {
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
        ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K,
                    ABLayout::MK, ABLayout::KN>(
        pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
        dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
    dYL1Buffer.UnlockCons();

    if constexpr (!ENABLE_UNITFLAG) {
        dvL0CBuffer.UnlockProd();
        dvL0CBuffer.LockCons();
    }
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = runInfo.shape.s2;
    fixpipeParams.nSize = realN;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = smallSDConstInfo.kvRowStride;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
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
    if constexpr (!ENABLE_UNITFLAG) {
        dvL0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF
class FAGBlockCubeSmallSDDummy {
public:
    __aicore__ inline FAGBlockCubeSmallSDDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, MutexBufferManager<BufferType::L1> *l1BuffMgr){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace){};
    template <typename InitInfo>
    __aicore__ inline void InitCubeBuffer(InitInfo &){};
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
