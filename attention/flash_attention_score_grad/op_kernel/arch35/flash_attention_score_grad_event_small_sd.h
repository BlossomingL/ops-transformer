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
 * \file flash_attention_score_grad_event_small_sd.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_EVENT_SMALL_SD_H_
#define FLASH_ATTENTION_SCORE_GRAD_EVENT_SMALL_SD_H_

#include <cstdint>

namespace commondef {
constexpr uint8_t SMALL_SD_CUBE_QK_READY_FLAG[2] = {0, 1};
constexpr uint8_t SMALL_SD_CUBE_DYV_READY_FLAG[2] = {2, 3};
constexpr uint8_t SMALL_SD_VECTOR_PDS_READY_FLAG[2] = {4, 5};
constexpr uint8_t SMALL_SD_SLOT_REUSE_READY_FLAG[2] = {6, 7};
constexpr uint8_t SMALL_SD_DS_L1_REUSABLE_FLAG = 8;
constexpr uint8_t SMALL_SD_P_L1_REUSABLE_FLAG = 9;
} // namespace commondef

#endif // FLASH_ATTENTION_SCORE_GRAD_EVENT_SMALL_SD_H_
