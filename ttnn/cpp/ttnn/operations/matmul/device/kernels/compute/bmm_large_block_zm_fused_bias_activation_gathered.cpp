// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "mod_div_lib.h"

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "remote_circular_buffer_api.h"
// #include "debug/dprint.h"

namespace NAMESPACE {

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr;
}

FORCE_INLINE void update_local_cb_rd_ptr(uint32_t cb_id, uint32_t val) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    local_cb.fifo_rd_ptr = val;
}

FORCE_INLINE uint32_t get_local_cb_start_addr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    uint32_t fifo_limit = local_cb.fifo_limit;
    uint32_t fifo_start_addr = fifo_limit - fifo_size;
    return fifo_start_addr;
}

FORCE_INLINE bool is_tensor_split(uint32_t cb_id, uint32_t tensor_size_bytes) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_limit = local_cb.fifo_limit;
    bool split = (fifo_limit - fifo_rd_ptr) < tensor_size_bytes / L1_ALIGNMENT;
    return split;
}

FORCE_INLINE void calculate_next_block_index_and_update_rd_ptr(
    uint32_t cb_id,
    uint32_t num_blocks,
    uint32_t block_size_bytes,
    uint32_t curr_block_index,
    uint32_t cb_start_addr,
    uint32_t rd_ptr_start_addr,
    bool tensor_split,
    uint32_t* updated_block_index,
    uint32_t* updated_rd_ptr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t next_block_index = curr_block_index + 1;
    uint32_t next_fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t block_size_bytes_aligned = block_size_bytes / L1_ALIGNMENT;
    bool reach_limit = local_cb.fifo_rd_ptr == local_cb.fifo_limit;
    bool last_block = curr_block_index == (num_blocks - 1);
    if (tensor_split) {
        if (reach_limit) {
            local_cb.fifo_rd_ptr = cb_start_addr;
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr = cb_start_addr + block_size_bytes_aligned;
            }
        } else {
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr += block_size_bytes_aligned;
            }
        }
    } else {
        if (last_block) {
            next_block_index = 0;
            next_fifo_rd_ptr = rd_ptr_start_addr;
        } else {
            next_fifo_rd_ptr += block_size_bytes_aligned;
        }
    }
    *updated_block_index = next_block_index;
    *updated_rd_ptr = next_fifo_rd_ptr;
}

FORCE_INLINE void update_rd_ptr_to_ring_index(
    uint32_t cb_id, uint32_t block_size_bytes, uint32_t ring_index, bool tensor_split) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    if (tensor_split) {
        if ((local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT) >= local_cb.fifo_limit) {
            uint32_t fifo_size = local_cb.fifo_size;
            uint32_t fifo_limit = local_cb.fifo_limit;
            uint32_t fifo_start_addr = fifo_limit - fifo_size;
            uint32_t fifo_size_skip_bytes = local_cb.fifo_rd_ptr - fifo_start_addr;
            local_cb.fifo_rd_ptr =
                fifo_start_addr +
                (fifo_size_skip_bytes + ring_index * block_size_bytes / L1_ALIGNMENT) % local_cb.fifo_size;

        } else {
            local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
        }
    } else {
        local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
    }
}

void MAIN {
    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                                  // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);      // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);  // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);  // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(11);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(12);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(13);                // untilize output
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(14);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t in2_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t sync_cb = tt::CBIndex::c_3;
    constexpr uint32_t sync_cb2 = tt::CBIndex::c_4;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_6;

    constexpr uint32_t mm_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;

    uint32_t in1_cb_start_addr = 0;
    uint32_t in1_rd_ptr_start_addr = 0;
    uint32_t curr_in1_block_index = 0;
    bool in1_tensor_split = 0;
    uint32_t next_in1_block_index;
    uint32_t next_in1_rd_ptr_addr;

    UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
    UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
    UNPACK((curr_in1_block_index = ring_idx));
    UNPACK((in1_tensor_split = is_tensor_split(in1_cb_id, in1_block_num_tiles * num_blocks * in1_tile_size)));

    UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_num_tiles * in1_tile_size, ring_idx, in1_tensor_split)));

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks > 1 && (out_block_num_tiles / out_subblock_num_tiles) > 1;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

#ifdef PACK_RELU
        // for each batch we start we relu disabled so that intermediate results are not relu'd
        if constexpr (batch > 1) {
            PACK((llk_pack_relu_config(ReluType::NO_RELU)));
        }
#endif

        if constexpr (batch > 1) {
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
        }

        // cb_wait_front(in1_cb_id, in1_block_num_tiles * num_blocks);
        cb_wait_front(sync_cb2, 1);
        cb_pop_front(sync_cb2, 1);

        // cb_pop_front(in1_cb_id, in1_block_num_tiles * ring_idx);
        for (uint32_t block = 0; block < num_blocks; block++) {
            const uint32_t input0_cb_id = block == 0 ? in0_cb_id : in2_cb_id;
            bool last_out = block == (num_blocks - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
            if (last_out) {
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
            }
#endif

            if (block == 0) {
                cb_reserve_back(input0_cb_id, in0_block_num_tiles);
                cb_push_back(input0_cb_id, in0_block_num_tiles);
            }

            cb_wait_front(input0_cb_id, in0_block_num_tiles);

            UNPACK((calculate_next_block_index_and_update_rd_ptr(
                in1_cb_id,
                num_blocks,
                in1_block_num_tiles * in1_tile_size,
                curr_in1_block_index,
                in1_cb_start_addr,
                in1_rd_ptr_start_addr,
                in1_tensor_split,
                &next_in1_block_index,
                &next_in1_rd_ptr_addr)));

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();
                    if (enable_reload) {
                        reload_from_cb_to_dst(
                            input0_cb_id,
                            in1_cb_id,
                            mm_partials_cb_id,
                            in1_transpose_tile,
                            out_subblock_num_tiles,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                    }

#ifndef SKIP_COMPUTE
                    // Compute output sub-block
                    uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                    uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                    uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                    // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                        // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                        // accumulation is done by iterating matmul_block across inner dim
                        // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                        matmul_block(
                            input0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            in1_transpose_tile,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        in0_index++;                  // stride right by 1
                        in1_index += in1_per_core_w;  // to stride down by 1 need to stride by in_per_core_w (should be
                                                      // called in1_block_w)
                    }

#endif  // SKIP_COMPUTE

                    if (last_out) {
// If we fuse bias, we will pack out and run bias + optional sfpu in a separate loop
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
#endif

                        tile_regs_commit();
                        // Pack out to output buffer
                        cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();

                        // Release in1
                        cb_reserve_back(sync_cb, 1);
                        cb_push_back(sync_cb, 1);
                        cb_pop_front(in1_cb_id, in1_block_num_tiles * num_blocks);

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                        PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC

                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                        uint32_t start_dst_index = 0;
                        matmul_pack_tile(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

                        tile_regs_release();
                        cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

                    } else if (spill) {
                        tile_regs_commit();
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();

#ifdef PACKER_L1_ACC
                        if (block == 0) {  // no accumulation for first iteration
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        } else if (block == 1) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
#endif

                        uint32_t start_dst_index = 0;
                        matmul_pack_tile(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

                        tile_regs_release();
                        cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

#ifdef PACKER_L1_ACC

            // Last iteration does spill and reload to output buffer
            if (block < num_blocks - 2 && spill) {
                cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
            }
            if (block == num_blocks - 2 && spill) {
                enable_reload = true;
            }  // reload when last iteration
#else
            if constexpr (spill) {
                enable_reload = true;
            }
#endif

            cb_pop_front(input0_cb_id, in0_block_num_tiles);
            curr_in1_block_index = next_in1_block_index;
            UNPACK((update_local_cb_rd_ptr(in1_cb_id, next_in1_rd_ptr_addr)));
        }

        if constexpr (batch > 1) {
            // reconfigure init for matmul
            mm_block_init_short(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

            // reconfigure unpacker df for src A
            reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
        }
    }
}
}  // namespace NAMESPACE
