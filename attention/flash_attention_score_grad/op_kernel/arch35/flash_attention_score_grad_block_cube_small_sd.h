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

#include "flash_attention_score_grad_block_cube.h"

namespace FagBaseApi {

TEMPLATES_DEF
class FAGBlockCubeSmallSD : public FAGBlockCube<TEMPLATE_ARGS> {
public:
    using BaseClass = FAGBlockCube<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockCubeSmallSD(){};
};

TEMPLATES_DEF
class FAGBlockCubeSmallSDDummy : public FAGBlockCubeDummy<TEMPLATE_ARGS> {
public:
    using BaseClass = FAGBlockCubeDummy<TEMPLATE_ARGS>;
    __aicore__ inline FAGBlockCubeSmallSDDummy(){};
};

DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeSmallSD);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeSmallSDDummy);

} // namespace FagBaseApi
#endif
