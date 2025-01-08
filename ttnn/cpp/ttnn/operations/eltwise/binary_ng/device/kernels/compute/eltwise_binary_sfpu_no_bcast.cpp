// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api/add_int32_sfpu.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS))
    BINARY_SFPU_INIT
#endif

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, onetile);
        cb_wait_front(cb_post_lhs, onetile);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, onetile);
        cb_wait_front(cb_post_rhs, onetile);

        cb_reserve_back(cb_out, onetile);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            BINARY_SFPU_OP(i * 2, i * 2 + 1);
            PROCESS_POST_ACTIVATIONS(i * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(i * 2, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_post_lhs, onetile);
        cb_pop_front(cb_post_rhs, onetile);
    }
}
}  // namespace NAMESPACE
