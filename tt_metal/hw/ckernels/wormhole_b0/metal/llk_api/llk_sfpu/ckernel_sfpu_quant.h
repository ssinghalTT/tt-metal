// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_quant_int32(const uint dst_offset)
{
    _quant_int32_<APPROXIMATION_MODE, ITERATIONS>(dst_offset);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_requant_int32(const uint dst_offset)
{
    _requant_int32_<APPROXIMATION_MODE, ITERATIONS>(dst_offset);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dequant_int32(const uint dst_offset)
{
    _dequant_int32_<APPROXIMATION_MODE, ITERATIONS>(dst_offset);
}

template <bool APPROXIMATION_MODE>
void quant_init(const uint zero_point) {
    _init_quant_zero_point_<APPROXIMATION_MODE>(zero_point);
}

}  // namespace sfpu
}  // namespace ckernel
