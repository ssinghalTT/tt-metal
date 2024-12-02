// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

enum { ADD = 0, SUB = 1, MUL = 2, DIV = 3, RSUB = 4 };

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
void calculate_binop_with_scalar(uint32_t param) {
    const vFloat parameter = Converter::to_float(param);

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat result = 0.0f;

        if constexpr (BINOP_MODE == ADD) {
            result = val + parameter;
        } else if constexpr (BINOP_MODE == SUB) {
            result = val - parameter;
        } else if constexpr (BINOP_MODE == MUL) {
            result = val * parameter;
        } else if constexpr (BINOP_MODE == DIV) {
            // parameter is inverted from host side and passed down
            result = val * parameter;
        } else if constexpr (BINOP_MODE == RSUB) {
            result = parameter - val;
        }
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
void calculate_add(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, ADD, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
void calculate_sub(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, SUB, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
void calculate_mul(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, MUL, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
void calculate_div(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, DIV, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
void calculate_rsub(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, RSUB, ITERATIONS>(param);
    return;
}

}  // namespace sfpu
}  // namespace ckernel
