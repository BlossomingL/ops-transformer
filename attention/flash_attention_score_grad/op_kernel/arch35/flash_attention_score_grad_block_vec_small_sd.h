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

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_common_small_sd.h"
#include "cube_api/mutex_buffer.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"

namespace FagBaseApi {

TEMPLATES_DEF
class FAGBlockVecSmallSD {
public:
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    __aicore__ inline FAGBlockVecSmallSD(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, uint32_t vSubBlockIdx);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace);
    __aicore__ inline void InitUbBuffer();
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
    uint32_t vSubBlockIdxForCompat = 0;
    TPipe *pipe;
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<OUTDTYPE> yGm, dyGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
};

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::SetVecBlockParams(
    TPipe *pipe, uint32_t vSubBlockIdx)
{
    vSubBlockIdxForCompat = vSubBlockIdx;
    this->pipe = pipe;
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR softmaxMax,
    GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace)
{
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::InitUbBuffer()
{
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    pipe->InitBuffer(dSOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(pOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec1SmallSD(
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    if (runInfo.shape.halfS1 == 0) {
        return;
    }
    const uint32_t dAlignToBlock = AlignTo(static_cast<uint32_t>(smallSDConstInfo.dv),
                                           INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    const uint32_t dstBlockStride = (HEAD_DIM_ALIGN - dAlignToBlock) * sizeof(OUTDTYPE) / 32;
    const uint32_t transposeStride = (smallSDConstInfo.qRowStride - smallSDConstInfo.dv) * sizeof(OUTDTYPE);
    const int64_t srcOffset =
        runInfo.offsets.dy + vSubBlockIdxForCompat * runInfo.shape.firstHalfS1 * smallSDConstInfo.qRowStride;

    LocalTensor<OUTDTYPE> yTensor = attenMaskOrYInQue.template AllocTensor<OUTDTYPE>();
    LocalTensor<OUTDTYPE> dyTensor = pseOrDyInQue.template AllocTensor<OUTDTYPE>();
    DataCopyPad(dyTensor, dyGm[srcOffset],
                {static_cast<uint16_t>(runInfo.shape.halfS1),
                 static_cast<uint32_t>(smallSDConstInfo.dv * sizeof(OUTDTYPE)),
                 transposeStride, dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>(dAlignToBlock - smallSDConstInfo.dv), 0});
    DataCopyPad(yTensor, yGm[srcOffset],
                {static_cast<uint16_t>(runInfo.shape.halfS1),
                 static_cast<uint32_t>(smallSDConstInfo.dv * sizeof(OUTDTYPE)),
                 transposeStride, dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>(dAlignToBlock - smallSDConstInfo.dv), 0});
    attenMaskOrYInQue.EnQue(yTensor);
    pseOrDyInQue.EnQue(dyTensor);

    yTensor = attenMaskOrYInQue.template DeQue<OUTDTYPE>();
    dyTensor = pseOrDyInQue.template DeQue<OUTDTYPE>();
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.template Get<CALC_TYPE>();
    AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
        softmaxGradResTensor, yTensor, dyTensor, runInfo.shape.halfS1, dAlignToBlock);
    attenMaskOrYInQue.FreeTensor(yTensor);
    pseOrDyInQue.FreeTensor(dyTensor);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::CopyMaxSumSmallSD(
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo, int64_t taskId)
{
    if (runInfo.shape.halfS1 == 0) {
        return;
    }
    constexpr uint32_t MAX_SUM_ELEMENT_COUNT = MAX_SUM_REDUCE_AXIS_SIZE / sizeof(float);
    int64_t maxSumGmOffset = 0;
    if (smallSDConstInfo.tndMaxSumLayout == MAX_SUM_TND) {
        maxSumGmOffset =
            (runInfo.offsets.q / smallSDConstInfo.d +
             vSubBlockIdxForCompat * runInfo.shape.firstHalfS1 * smallSDConstInfo.n2Size) *
            MAX_SUM_ELEMENT_COUNT;
    } else if (smallSDConstInfo.layoutType == TND) {
        maxSumGmOffset =
            (runInfo.qPrefix * smallSDConstInfo.n2Size + runInfo.n2Idx * runInfo.shape.s1 +
             vSubBlockIdxForCompat * runInfo.shape.firstHalfS1) *
            MAX_SUM_ELEMENT_COUNT;
    } else {
        maxSumGmOffset =
            ((runInfo.batchIdx * smallSDConstInfo.n2Size + runInfo.n2Idx) * smallSDConstInfo.s1 +
             vSubBlockIdxForCompat * runInfo.shape.firstHalfS1) *
            MAX_SUM_ELEMENT_COUNT;
    }

    LocalTensor<float> maxSumTensor = maxSumQue[taskId & 1].template AllocTensor<float>();
    if (smallSDConstInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = smallSDConstInfo.n2Size * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, softmaxSumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.shape.halfS1),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_ELEMENT_COUNT], softmaxMaxGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.shape.halfS1),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, softmaxSumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.shape.halfS1 * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_ELEMENT_COUNT], softmaxMaxGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.shape.halfS1 * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
    }
    maxSumQue[taskId & 1].EnQue(maxSumTensor);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec2SmallSD(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    if (runInfo.shape.halfS1 == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<OUTDTYPE> pseTensor;
    LocalTensor<CALC_TYPE> maxSumTensor = maxSumQue[runInfo.taskId & 1].template DeQue<CALC_TYPE>();
    if (runInfo.shape.s2 > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
        AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned128),
                                      false, false, false>(
            mm2ResTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            mm2ResTensor, pseTensor, attenMaskTensor, smallSDConstInfo.scaleValue, 0, runInfo.shape.halfS1,
            runInfo.shape.s2);
    } else {
        AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64),
                                      false, false, false>(
            mm2ResTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            mm2ResTensor, pseTensor, attenMaskTensor, smallSDConstInfo.scaleValue, 0, runInfo.shape.halfS1,
            runInfo.shape.s2);
    }
    maxSumQue[runInfo.taskId & 1].FreeTensor(maxSumTensor);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec3SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    if (runInfo.shape.halfS1 == 0) {
        return;
    }

    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.template Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.template AllocTensor<INPUT_TYPE>();
    if (runInfo.shape.s2 > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
        BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned128), 0, false>(
            mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.shape.halfS1, runInfo.shape.s2);
    } else {
        BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64), false, false>(
            mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.shape.halfS1, runInfo.shape.s2);
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.template DeQue<INPUT_TYPE>();

    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.template GetTensor<INPUT_TYPE>();
    uint32_t scmOffset =
        vSubBlockIdxForCompat == 0 ? 0 : runInfo.shape.firstHalfS1 * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    dataCopyParams.blockLen = static_cast<uint16_t>(
        runInfo.shape.halfS1 * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    dataCopyParams.srcStride = static_cast<uint16_t>(
        (VECTOR_BASEM + 1 - runInfo.shape.halfS1) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE /
        INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.shape.s1);
    dataCopyParams.dstStride =
        (s1RealSizeAlignTo16 - runInfo.shape.halfS1) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE /
        INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
    DataCopy(dsL1Tensor[scmOffset], vecOutBuffer, dataCopyParams);
    dSOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec4SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
    const SmallSDConstInfo &smallSDConstInfo, const SmallSDRunInfo &runInfo)
{
    if (runInfo.shape.halfS1 == 0) {
        return;
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<OUTDTYPE> vecOutBuffer = pOutQue.template AllocTensor<OUTDTYPE>();
    CastTransdataDeconflict<OUTDTYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm2ResTensor, selrIndexesTensor,
                                                               VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer);
    pOutQue.template DeQue<OUTDTYPE>();

    LocalTensor<OUTDTYPE> pL1Tensor = dstBuffer.template GetTensor<OUTDTYPE>();
    uint32_t scmOffset =
        vSubBlockIdxForCompat == 0 ? 0 : runInfo.shape.firstHalfS1 * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
    dataCopyParams.blockLen = static_cast<uint16_t>(
        runInfo.shape.halfS1 * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE);
    dataCopyParams.srcStride = static_cast<uint16_t>(
        (VECTOR_BASEM + 1 - runInfo.shape.halfS1) * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE /
        INPUT_BLOCK_NUM_FOR_OUT_DTYPE);
    uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.shape.s1);
    dataCopyParams.dstStride =
        (s1RealSizeAlignTo16 - runInfo.shape.halfS1) * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE /
        INPUT_BLOCK_NUM_FOR_OUT_DTYPE;
    DataCopy(pL1Tensor[scmOffset], vecOutBuffer, dataCopyParams);
    pOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessMulsAndCastSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor, const SmallSDConstInfo &smallSDConstInfo,
    const SmallSDRunInfo &runInfo)
{
    if constexpr (IS_WRITE_UB) {
        if ((MM_IDX == DQ_IDX && runInfo.shape.halfS1 == 0) || (MM_IDX != DQ_IDX && runInfo.shape.halfS2 == 0)) {
            return;
        }

        uint64_t dSize = (MM_IDX == DV_IDX) ? smallSDConstInfo.dv : smallSDConstInfo.d;
        DataCopyExtParams intriParamsOut;
        intriParamsOut.blockCount =
            (MM_IDX == DQ_IDX) ? static_cast<uint16_t>(runInfo.shape.halfS1) :
                                 static_cast<uint16_t>(runInfo.shape.halfS2);
        intriParamsOut.blockLen = static_cast<uint32_t>(dSize * sizeof(OUTDTYPE));
        intriParamsOut.srcStride = 0;

        uint64_t rowStride = (MM_IDX == DQ_IDX) ? smallSDConstInfo.qRowStride : smallSDConstInfo.kvRowStride;
        intriParamsOut.dstStride = static_cast<uint32_t>((rowStride - dSize) * sizeof(OUTDTYPE));
        uint64_t halfSRealSize = (MM_IDX == DQ_IDX) ? runInfo.shape.firstHalfS1 : runInfo.shape.firstHalfS2;
        uint64_t dqkvGmOffset =
            (MM_IDX == DQ_IDX) ? runInfo.offsets.dq : (MM_IDX == DK_IDX ? runInfo.offsets.dk : runInfo.offsets.dv);
        dqkvGmOffset += vSubBlockIdxForCompat * halfSRealSize * rowStride;

        uint32_t dataSize = intriParamsOut.blockCount * AlignTo16(dSize);
        if constexpr (MM_IDX != DV_IDX) {
            Muls(inputTensor, inputTensor, smallSDConstInfo.scaleValue, dataSize);
        }

        LocalTensor<OUTDTYPE> dqkvCastTensor = dSOutQue.template AllocTensor<OUTDTYPE>();
        Cast(dqkvCastTensor, inputTensor, RoundMode::CAST_ROUND, dataSize);
        dSOutQue.EnQue(dqkvCastTensor);
        dSOutQue.template DeQue<OUTDTYPE>();

        GlobalTensor<OUTDTYPE> dqkvGmTensor = MM_IDX == DQ_IDX ? dqGm : (MM_IDX == DK_IDX ? dkGm : dvGm);
        DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
        dSOutQue.FreeTensor(dqkvCastTensor);
    }
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
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, uint32_t vSubBlockIdx){};
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
