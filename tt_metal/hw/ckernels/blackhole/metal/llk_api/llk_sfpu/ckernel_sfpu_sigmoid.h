// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL5(coef4, coef3, coef2, coef1, coef0, val) \
    ((((coef4 * val + coef3) * val + coef2) * val + coef1) * val + coef0)

inline vFloat sigmoid_piecewise_linear_positive(vFloat val) {
    vFloat result = 0.0f;
    v_if(val >= +5.0f) { result = 1.0f; }
    v_elseif(val > 1.0f && val < 5.0f) {
        result = POLYVAL5(0.00144462f, -0.01055479f, -0.01203685f, 0.24300185f, 0.50437757f, val);
    }
    v_else {
        result = 0.229f * val + 0.5f;  // linear appx as y = 0.229x + 0.5
    }
    v_endif;
    return result;
}

// sigmoid is anti-symmetric and offset by 1
// sigmoid[-x] = 1 - sigmoid[x]
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat result = 0.0f;

        v_if(val < 0.0f) { val = -val; }
        v_endif;

        result = sigmoid_piecewise_linear_positive(val);

        val = dst_reg[0];
        v_if(val < 0.0f) { result = 1.0f - result; }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }

    return;
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    // imm0 = 0x3DFF;
    // imm1 = 0x21D8;
    // imm2 = 0xFF10;
    // TTI_SFPLOADI(0, 2, imm0);
    // TTI_SFPLOADI(1, 2, imm1);
    // TTI_SFPLOADI(2, 2, imm2);
    // Using a 6 piece LUT to calculate and model sigmoid  directly
    // x <= 0.5 --> 0.2452x + (-0.0004997)
    // x <= 1.0 --> 0.2173x + 0.0152
    // x <= 1.5 --> 0.1731x + 0.05988
    // x <= 2.0 --> 0.1262x + 0.1298
    // x <= 4.0 --> 0.0485x + 0.2998
    // x >  4.0 --> 0.4998

    // imm0[15:0] = A0=0.2452 = 0x33D9 -- imm0[31:16] = A1=0.2173 = 0x32F4
    _sfpu_load_imm32_(0, 0x32F433D9);
    // imm4[15:0] = B0= -0.0004997  = 0x9018 -- imm4[31:16] = B1= 0.0152 = 0x23c8
    _sfpu_load_imm32_(4, 0x23C89018);

    // imm1[15:0] = A2=0.1731 = 0x318a -- imm1[31:16] = A3=0.1262 = 0x300a
    _sfpu_load_imm32_(1, 0x300A318A);
    // imm5[15:0] = B2=0.05988 = 0x2BAA -- imm5[31:16] = B3=0.1298 = 0x3027
    _sfpu_load_imm32_(5, 0x30272BAA);

    // imm2[15:0] = A4=0.0485 = 0x2A35 -- imm2[31:16] = A5=0.0 = 0x7C00
    _sfpu_load_imm32_(2, 0x7C002A35);
    // imm6[15:0] = B4=0.2998 = 0x34CC -- imm6[31:16] = B5=0.4998 = 0x37ff
    _sfpu_load_imm32_(6, 0x37ff34CC);
}

}  // namespace sfpu
}  // namespace ckernel
