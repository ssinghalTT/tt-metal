// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);
    uint32_t n_stride = get_arg_val<uint32_t>(5);
    uint32_t c_stride = get_arg_val<uint32_t>(6);
    uint32_t N = get_arg_val<uint32_t>(7);
    uint32_t C = get_arg_val<uint32_t>(8);
    uint32_t Ht = get_arg_val<uint32_t>(9);
    uint32_t Wt = get_arg_val<uint32_t>(10);

    constexpr uint32_t onetile = 1;

    constexpr auto cb_id_src = tt::CBIndex::c_1;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);

    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const DataFormat dst_data_format = get_dataformat(cb_id_dst);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;
    uint32_t start_th = start_t / Wt;
    uint32_t start_tw = start_t % Wt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_written < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_written < num_tiles; ++tw, ++num_tiles_written) {
                    // read a tile from src
                    cb_reserve_back(cb_id_src, onetile);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_src);
                    noc_async_read_tile(tile_offset + tw, src, l1_write_addr);
                    noc_async_read_barrier();
                    FILL_TILE_WITH_FIRST_ROW(cb_id_src);
                    cb_push_back(cb_id_src, onetile);

                    // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                    cb_wait_front(cb_id_dst, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                    noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_dst, onetile);
                }
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
