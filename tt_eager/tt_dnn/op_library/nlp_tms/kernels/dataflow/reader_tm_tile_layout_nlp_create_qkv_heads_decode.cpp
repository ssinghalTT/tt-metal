// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {

    uint32_t head_size                     = get_arg_val<uint32_t>(0);
    uint32_t num_q_heads                   = get_arg_val<uint32_t>(1);
    uint32_t num_kv_heads                   = get_arg_val<uint32_t>(2);
    uint32_t head_size_num_tiles           = get_arg_val<uint32_t>(3);
    uint32_t in_tile_offset_by_batch       = get_arg_val<uint32_t>(4);
    uint32_t start_qkv_x                     = get_arg_val<uint32_t>(5);
    uint32_t start_qkv_y                     = get_arg_val<uint32_t>(6);
    uint32_t q_start_addr                  = get_arg_val<uint32_t>(7);
    uint32_t k_start_addr                  = get_arg_val<uint32_t>(8);
    uint32_t v_start_addr                  = get_arg_val<uint32_t>(9);

    constexpr uint32_t ELEMENT_SIZE        = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES  = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out         = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_k_out         = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_v_out         = get_compile_time_arg_val(4);

    uint32_t num_x                         = get_arg_val<uint32_t>(10);
    uint32_t num_y                         = get_arg_val<uint32_t>(11);
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12));
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_x));

    // Q
    uint32_t qkv_x = start_qkv_x;
    uint32_t qkv_y = start_qkv_y;
    uint32_t total_input_cores = num_x * num_y;
    uint32_t num_tiles_per_core = head_size_num_tiles * (num_q_heads + 2 * num_kv_heads) / total_input_cores;

    // // debug for loop
    // DPRINT << "[mikevin DPRINT] NOC coordinates:" << ENDL();
    // DPRINT << "     total_input_cores: " << total_input_cores << ENDL();
    // DPRINT << "     num_tiles_per_core: " << num_tiles_per_core << ENDL();
    // for (uint32_t i = 0; i < num_x; i++){
    //     for (uint32_t j = 0; j < num_y; j++){
    //         DPRINT << "         " << in0_mcast_noc_x[i] << ", " << in0_mcast_noc_y[j] << ENDL();
    //     }
    // }


    uint64_t qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_batch;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size/head_size_num_tiles;

    // DPRINT << "[xuncai DPRINT] head_size = " << head_size << ENDL();
    // DPRINT << "[xuncai DPRINT] head_size_num_tiles = " << head_size_num_tiles << ENDL();
    // DPRINT << "[xuncai DPRINT] ELEMENT_SIZE = " << ELEMENT_SIZE << ENDL();
    // DPRINT << "[xuncai DPRINT] SUBTILE_LINE_BYTES = " << SUBTILE_LINE_BYTES << ENDL();
    // DPRINT << "[xuncai DPRINT] tile_size = " << tile_size << ENDL();
    // DPRINT << "[xuncai DPRINT] num_q_heads = " << num_q_heads << ENDL();
    // DPRINT << "[xuncai DPRINT] num_kv_heads = " << num_kv_heads << ENDL();
    // DPRINT << "[xuncai DPRINT] in_tile_offset_by_batch = " << in_tile_offset_by_batch << ENDL();

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    // DPRINT << "[xuncai DPRINT] Q read:" << ENDL();
    // DPRINT << "         qkv_read_addr = " << qkv_read_addr << ENDL();
    // DPRINT << "         q_write_addr = " << q_write_addr << ENDL();
    for (uint32_t q = 0; q < num_q_heads; ++q) {
        // DPRINT << "[xuncai DPRINT] q = " << q << ENDL();
        uint32_t wptr_offset = q < 16 ? q * SUBTILE_LINE_BYTES : (q - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t q_write_addr = get_write_ptr(cb_id_q_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // DPRINT << "[xuncai DPRINT] i = " << i << ENDL();
            // DPRINT << "         qkv_read_addr = " << qkv_read_addr << ENDL();
            // DPRINT << "         q_write_addr = " << q_write_addr << ENDL();
            // Read first phase
            noc_async_read(qkv_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();
            // Read second phase
            noc_async_read(qkv_read_addr+256*ELEMENT_SIZE, q_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                qkv_x++;
                if (qkv_x == num_x) {
                    qkv_x = 0;
                    qkv_y++;
                }
                qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_batch;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    // K
    uint32_t k_write_addr = 0;

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    for (uint32_t k = 0; k < num_kv_heads; ++k) {
        uint32_t wptr_offset = k < 16 ? k * SUBTILE_LINE_BYTES : (k - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t k_write_addr = get_write_ptr(cb_id_k_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            noc_async_read(qkv_read_addr, k_write_addr, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();
            // Read second phase
            noc_async_read(qkv_read_addr+256*ELEMENT_SIZE, k_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();

            qkv_read_addr += tile_size;
            k_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                qkv_x++;
                if (qkv_x == num_x) {
                    qkv_x = 0;
                    qkv_y++;
                }
                qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_batch;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    // v
    uint32_t v_write_addr = 0;

    // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
    for (uint32_t v = 0; v < num_kv_heads; ++v) {
        uint32_t wptr_offset = v < 16 ? v * SUBTILE_LINE_BYTES : (v - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t v_write_addr = get_write_ptr(cb_id_v_out) + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            noc_async_read(qkv_read_addr, v_write_addr, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();
            // Read second phase
            noc_async_read(qkv_read_addr+256*ELEMENT_SIZE, v_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            //noc_async_read_barrier();

            qkv_read_addr += tile_size;
            v_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                qkv_x++;
                if (qkv_x == num_x) {
                    qkv_x = 0;
                    qkv_y++;
                }
                qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_batch;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc_async_read_barrier();
}
