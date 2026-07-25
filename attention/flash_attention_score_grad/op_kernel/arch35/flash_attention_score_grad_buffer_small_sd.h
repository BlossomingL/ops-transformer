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
 * \file flash_attention_score_grad_buffer_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BUFFER_SMALL_SD_H_
#define FLASH_ATTENTION_SCORE_GRAD_BUFFER_SMALL_SD_H_

#include <cstdint>
#include "flash_attention_score_grad_common.h"

namespace FagSmallSDApi {
template <uint32_t HeadDim> struct SmallSDBufferLayout {
    static_assert(HeadDim == 64 || HeadDim == 128, "SmallSD only supports HeadDim 64/128.");
    static constexpr uint32_t BASE_M = 128;
    static constexpr uint32_t BASE_N = 128;
    static constexpr uint32_t INPUT_TILE_ELEMS = BASE_M * HeadDim;
    static constexpr uint32_t SCORE_TILE_ELEMS = BASE_M * BASE_N;
    static constexpr uint32_t L1_Q_OFFSET = 0;
    static constexpr uint32_t L1_DY_OFFSET = L1_Q_OFFSET + INPUT_TILE_ELEMS;
    static constexpr uint32_t L1_K_OFFSET = L1_DY_OFFSET + INPUT_TILE_ELEMS;
    static constexpr uint32_t L1_V_OFFSET = L1_K_OFFSET + INPUT_TILE_ELEMS;
    static constexpr uint32_t L1_P_OFFSET = L1_V_OFFSET + INPUT_TILE_ELEMS;
    static constexpr uint32_t L1_DS_OFFSET = L1_P_OFFSET + SCORE_TILE_ELEMS;
    static constexpr uint32_t L1_TOTAL_ELEMS = L1_DS_OFFSET + SCORE_TILE_ELEMS;
    static constexpr uint32_t L0A_PING_OFFSET = 0;
    static constexpr uint32_t L0A_PONG_OFFSET = INPUT_TILE_ELEMS;
    static constexpr uint32_t L0B_PING_OFFSET = 0;
    static constexpr uint32_t L0B_PONG_OFFSET = INPUT_TILE_ELEMS;
};
} // namespace FagSmallSDApi

#endif // FLASH_ATTENTION_SCORE_GRAD_BUFFER_SMALL_SD_H_
