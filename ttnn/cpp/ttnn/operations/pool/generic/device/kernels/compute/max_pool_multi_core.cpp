// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "tt_metal/hw/inc/debug/dprint_tensix.h"

inline void print_page(uint32_t l1_addr, uint32_t pagelen) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t i = 0; i < pagelen; ++i) {
        DPRINT << BF16(*ptr) << " ";
        ptr++;
    }
    DPRINT << ENDL();
}

inline void print_page_i(uint32_t l1_addr, uint32_t pagelen, uint32_t page_index) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    ptr += pagelen * page_index;
    for (uint32_t i = 0; i < pagelen; ++i) {
        DPRINT << BF16(*ptr) << " ";
        ptr++;
    }
    DPRINT << ENDL();
}

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n/2; ++ i) {
        ptr[i] = (val | (val << 16));
    }
    return true;
}

template<uint32_t num_output_tiles, bool is_partial_tile, uint32_t split_reader, uint32_t unpA_face_r_dim>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id) {

    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    cb_reserve_back(out_cb_id, 1);
    PACK(DPRINT << "******************" << ENDL(););
    PACK(DPRINT << "BEFORE OUTPUT PAGE" << ENDL(););
    PACK(uint32_t out_l1_write_addr = CB_WR_PTR(out_cb_id));
    PACK(print_page(out_l1_write_addr, 256););
    PACK(DPRINT << "_________________" << ENDL(););

    const uint32_t curr_in_cb_id = split_reader ? (in_cb_id + (in_stick_index & 0x1)) : in_cb_id;
    cb_wait_front(curr_in_cb_id, 1);
    tile_regs_acquire();
    unpack_tilizeA_B_block(curr_in_cb_id, in_scalar_cb_id, num_output_tiles, 0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/, num_faces_in_tile /* unpack 1 or 2 faces ) */, unpA_face_r_dim);
    for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
        reduce_tile_math(c_i,  num_faces_in_tile /* reduce 1 or 2 faces */);
    }

    // The issue of finding infinities in the out_cb_id is not dependent on whether the tensix_sync is applied or a
    // large delay is applied It is verily possible that outputing too much data causes the dprint buffer to extend to
    // out_cb_id and when we write the negative infinities we are actually overwriting the out_cb_id. tensix_sync();
    // PACK(add_nops(););
    for (uint32_t x = 0; x < 8; x++) {
        dprint_tensix_dest_reg(x);
    }

    cb_pop_front(curr_in_cb_id, 1);
    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(out_cb_id, 1/*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile);  /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();

    // In the last iteration the data will be corrupted on wormhole_b0 for the debug print buffer.
    // We will have negative infinity as the garbage data, even though we overwrite it, debug print
    // Will print that negative infinities are there. But if we use the debug print from unpacker
    // At the end of the program, we will see the correct result.
    PACK(uint32_t out_l1_write_addr_a = CB_WR_PTR(out_cb_id));
    PACK(DPRINT << "******************" << ENDL(););
    PACK(DPRINT << "AFTER OUTPUT PAGE" << ENDL(););
    PACK(print_page(out_l1_write_addr_a, 256););
    PACK(DPRINT << "_________________" << ENDL(););

    cb_push_back(out_cb_id, 1);
}

namespace NAMESPACE {

void MAIN {

    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(3);
    constexpr uint32_t out_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);

    constexpr uint32_t split_reader = get_compile_time_arg_val(12);

    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t in_c = get_compile_time_arg_val(14);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(15);

    constexpr uint32_t in_cb_id = tt::CB::c_in0; // and tt::CB::c_in1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;
    constexpr uint32_t in_tiled_cb_id = tt::CB::c_intermed0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter = in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles = in_ntiles_c % MAX_TILES_PER_REDUCTION;
    tilizeA_B_reduce_init(in_cb_id,
                                in_scalar_cb_id,
                                max_tiles_per_iter,
                                out_cb_id,
                                num_faces_in_tile,
                                window_size_hw);
    pack_untilize_dst_init_short<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core; ++ i) {
        for (uint32_t b_i = 0; b_i < in_nblocks_c; ++ b_i) {
            if (b_i == in_nblocks_c - 1 && partial_iter_output_tiles > 0) {
                pack_untilize_uninit(out_cb_id);
                pack_untilize_dst_init_short<partial_iter_output_tiles>(out_cb_id, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
                reduce_h_fused<partial_iter_output_tiles, is_partial_tile, split_reader, window_size_hw>(in_cb_id, in_scalar_cb_id, i, out_cb_id);
            } else {
                pack_untilize_uninit(out_cb_id);
                pack_untilize_dst_init_short<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
                reduce_h_fused<max_tiles_per_iter, is_partial_tile, split_reader, window_size_hw>(in_cb_id, in_scalar_cb_id, i, out_cb_id);
            }
        }
    }
    cb_pop_front(in_scalar_cb_id, 1);

    // All the pages will show expected data, that is 8 pages, even ones containing 256 4s,
    // and odd ones containing 128 4s followed by 128 garbage data, which for the last one
    // Will be negative infinity.
    UNPACK(DPRINT << "UNPACKER PRINTING OUTPUT PAGES " << ENDL();)
    UNPACK(uint32_t out_l1_read_addr = CB_RD_PTR(out_cb_id););
    for (uint32_t x = 0; x < 8; x++) {
        UNPACK(print_page_i(out_l1_read_addr, 256, x);)
        UNPACK(DPRINT << ENDL(););
    }
}

}  // namespace NAMESPACE
