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

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H_
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H_

#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

namespace FagSmallSDApi {
template <typename InputType, typename CalcType, typename OutputType, bool IsTnd, uint32_t HeadDim, uint32_t Layout>
class SmallSDCubeBlock {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void IssueFirstStage(const SmallSDCoreTaskParamRegbase &, const SmallSDBaseParamRegbase &)
    {
    }

    __aicore__ inline void IssueGradStage(const SmallSDCoreTaskParamRegbase &, const SmallSDBaseParamRegbase &)
    {
    }

private:
    TPipe *pipe_ = nullptr;
};
} // namespace FagSmallSDApi

#endif // FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H_
