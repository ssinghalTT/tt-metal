// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/reduce.h"

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

template <
    uint32_t in_ntiles_c,
    uint32_t out_ntiles_c,
    bool is_partial_tile,
    uint32_t unpA_face_r_dim,
    uint32_t in_nblocks_c>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_ntiles_hwc_block,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id) {
    constexpr uint32_t num_output_tiles = out_ntiles_c / in_nblocks_c;
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;
    for (uint32_t c_i = 0; c_i < in_nblocks_c; ++c_i) {
        cb_reserve_back(out_cb_id, 1);
        const uint32_t curr_in_cb_id = in_cb_id;
        cb_wait_front(curr_in_cb_id, 1);
        tile_regs_acquire();

        unpack_tilizeA_B_block(
            curr_in_cb_id,
            in_scalar_cb_id,
            in_ntiles_hwc_block,
            0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
            num_faces_in_tile /* unpack 1 or 2 faces ) */,
            unpA_face_r_dim);
        for (uint32_t c_j = 0; c_j < in_ntiles_c / in_nblocks_c; ++c_j) {
            reduce_tile_math(c_j, num_faces_in_tile /* reduce 1 or 2 faces */);
        }

        cb_pop_front(curr_in_cb_id, 1);
        tile_regs_wait();
        tile_regs_commit();
        pack_untilize_dst<num_output_tiles>(
            out_cb_id, 1 /*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
        tile_regs_release();
        cb_push_back(out_cb_id, 1);
    }
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_hwc = get_compile_time_arg_val(1);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(2);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(3);
    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t in_c = get_compile_time_arg_val(5);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id = tt::CB::c_in0;  // and tt::CB::c_in1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t in_ntiles_hwc_block = in_ntiles_hwc / in_nblocks_c;
    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX/SUM");
    constexpr bool neginf_srca = (REDUCE_OP == PoolType::MAX ? true : false);
    constexpr bool zero_srca_reduce = (REDUCE_OP == PoolType::MAX ? false : true);

    tilizeA_B_reduce_init<neginf_srca, zero_srca_reduce>(
        in_cb_id, in_scalar_cb_id, in_ntiles_hwc_block, out_cb_id, num_faces_in_tile, window_size_hw);
    pack_untilize_dst_init_short<out_ntiles_c>(out_cb_id, num_out_rows, num_faces_in_tile);

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core; ++i) {
        reduce_h_fused<in_ntiles_c, out_ntiles_c, is_partial_tile, window_size_hw, in_nblocks_c>(
            in_cb_id, in_scalar_cb_id, in_ntiles_hwc_block, i, out_cb_id);
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
