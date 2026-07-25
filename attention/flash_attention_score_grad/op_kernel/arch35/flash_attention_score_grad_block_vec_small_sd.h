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
class FAGBlockVecSmallSD : public FAGBlockVec<TEMPLATE_ARGS> {
public:
    using BaseClass = FAGBlockVec<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockVecSmallSD(){};
    __aicore__ inline void ProcessVec1SmallSD(const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
                                             FagRunInfo &compatRunInfo);
    __aicore__ inline void CopyMaxSumSmallSD(const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
                                            FagRunInfo &compatRunInfo, int64_t taskId);
    __aicore__ inline void ProcessVec2SmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo);
    __aicore__ inline void ProcessVec3SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm1ResTensor,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo);
    __aicore__ inline void ProcessVec4SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo);
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCastSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                                    const SmallSDConstInfo &smallSDConstInfo,
                                                    FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo);
};

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec1SmallSD(
    const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo)
{
    (void)smallSDConstInfo;
    BaseClass::ProcessVec1(compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::CopyMaxSumSmallSD(
    const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo, int64_t taskId)
{
    (void)smallSDConstInfo;
    BaseClass::CopyMaxSum(compatConstInfo, compatRunInfo, taskId);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec2SmallSD(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
    FagRunInfo &compatRunInfo)
{
    (void)smallSDConstInfo;
    BaseClass::ProcessVec2(mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec3SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
    LocalTensor<CALC_TYPE> &mm2ResTensor, const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
    FagRunInfo &compatRunInfo)
{
    (void)smallSDConstInfo;
    BaseClass::ProcessVec3(dstBuffer, mm1ResTensor, mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessVec4SmallSD(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
    const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo)
{
    (void)smallSDConstInfo;
    BaseClass::ProcessVec4(dstBuffer, mm2ResTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
__aicore__ inline void FAGBlockVecSmallSD<TEMPLATE_ARGS>::ProcessMulsAndCastSmallSD(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor, const SmallSDConstInfo &smallSDConstInfo,
    FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo)
{
    (void)smallSDConstInfo;
    BaseClass::template ProcessMulsAndCast<T, IS_WRITE_UB, MM_IDX>(inputTensor, compatConstInfo, compatRunInfo);
}

TEMPLATES_DEF
class FAGBlockVecSmallSDDummy : public FAGBlockVecDummy<TEMPLATE_ARGS> {
public:
    using BaseClass = FAGBlockVecDummy<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockVecSmallSDDummy(){};
    __aicore__ inline void ProcessVec1SmallSD(const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
                                             FagRunInfo &compatRunInfo){};
    __aicore__ inline void CopyMaxSumSmallSD(const SmallSDConstInfo &smallSDConstInfo, FagConstInfo &compatConstInfo,
                                            FagRunInfo &compatRunInfo, int64_t taskId){};
    __aicore__ inline void ProcessVec2SmallSD(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo){};
    __aicore__ inline void ProcessVec3SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm1ResTensor,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo){};
    __aicore__ inline void ProcessVec4SmallSD(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                             LocalTensor<CALC_TYPE> &mm2ResTensor,
                                             const SmallSDConstInfo &smallSDConstInfo,
                                             FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo){};
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCastSmallSD(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                                    const SmallSDConstInfo &smallSDConstInfo,
                                                    FagConstInfo &compatConstInfo, FagRunInfo &compatRunInfo){};
};

} // namespace FagBaseApi
#endif
