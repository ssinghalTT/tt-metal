// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);        // batch_mean
    uint32_t batch_var_addr = get_arg_val<uint32_t>(1);  // batch_var
    const bool weight_has_value = get_arg_val<uint32_t>(2) == 1;
    uint32_t weight_addr = get_arg_val<uint32_t>(3);  // weight
    const bool bias_has_value = get_arg_val<uint32_t>(4) == 1;
    uint32_t bias_addr = get_arg_val<uint32_t>(5);           // bias
    const bool running_mean_has_value = get_arg_val<uint32_t>(6) == 1;
    uint32_t running_mean_addr = get_arg_val<uint32_t>(7);   // running_mean
    uint32_t dst_addr = get_arg_val<uint32_t>(8);            // output
    const bool is_training_mode = get_arg_val<uint32_t>(9);  // mode of operation
    uint32_t start_tile_id = get_arg_val<uint32_t>(10);
    uint32_t num_tiles = get_arg_val<uint32_t>(11);
    uint32_t HtWt = get_arg_val<uint32_t>(12);
    uint32_t n_stride = get_arg_val<uint32_t>(13);
    uint32_t c_stride = get_arg_val<uint32_t>(14);
    uint32_t N = get_arg_val<uint32_t>(15);
    uint32_t C = get_arg_val<uint32_t>(16);

    constexpr uint32_t onetile = 1;

    // batch_mean
    constexpr auto cb_id_src = tt::CBIndex::c_1;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);

    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    // output
    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const DataFormat dst_data_format = get_dataformat(cb_id_dst);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};

    // batch_var
    constexpr auto cb_id_batch_var = tt::CBIndex::c_3;
    constexpr bool batch_var_is_dram = get_compile_time_arg_val(2) == 1;
    const uint32_t batch_var_tile_bytes = get_tile_size(cb_id_batch_var);
    const DataFormat batch_var_data_format = get_dataformat(cb_id_batch_var);

    const InterleavedAddrGenFast<batch_var_is_dram> batch_var = {
        .bank_base_address = batch_var_addr, .page_size = batch_var_tile_bytes, .data_format = batch_var_data_format};

    // weight
    constexpr auto cb_id_weight = tt::CBIndex::c_16;
    constexpr bool weight_is_dram = get_compile_time_arg_val(3) == 1;
    const uint32_t weight_tile_bytes = get_tile_size(cb_id_weight);
    const DataFormat weight_data_format = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<weight_is_dram> weight = {
        .bank_base_address = weight_addr, .page_size = weight_tile_bytes, .data_format = weight_data_format};

    // bias
    constexpr auto cb_id_bias = tt::CBIndex::c_18;
    constexpr bool bias_is_dram = get_compile_time_arg_val(4) == 1;
    const uint32_t bias_tile_bytes = get_tile_size(cb_id_bias);
    const DataFormat bias_data_format = get_dataformat(cb_id_bias);

    const InterleavedAddrGenFast<bias_is_dram> bias = {
        .bank_base_address = bias_addr, .page_size = bias_tile_bytes, .data_format = bias_data_format};

    // running_mean
    constexpr auto cb_id_running_mean = tt::CBIndex::c_24;
    constexpr bool running_mean_is_dram = get_compile_time_arg_val(5) == 1;
    const uint32_t running_mean_tile_bytes = get_tile_size(cb_id_running_mean);
    const DataFormat running_mean_data_format = get_dataformat(cb_id_running_mean);

    const InterleavedAddrGenFast<running_mean_is_dram> running_mean = {
        .bank_base_address = running_mean_addr,
        .page_size = running_mean_tile_bytes,
        .data_format = running_mean_data_format};

    // write updated running stats
    constexpr auto cb_id_updated_running_mean = tt::CBIndex::c_25;

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_t = 0) {
            // read a tile from src
            cb_reserve_back(cb_id_src, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_src);
            noc_async_read_tile(tile_offset, src, l1_write_addr);
            noc_async_read_barrier();
            fill_tile_with_first_element_bfloat16(cb_id_src);
            cb_push_back(cb_id_src, onetile);

            // read a tile from batch variance
            cb_reserve_back(cb_id_batch_var, onetile);
            uint32_t l1_batch_var_write_addr = get_write_ptr(cb_id_batch_var);
            noc_async_read_tile(tile_offset, batch_var, l1_batch_var_write_addr);
            noc_async_read_barrier();
            fill_tile_with_first_element_bfloat16(cb_id_batch_var);
            cb_push_back(cb_id_batch_var, onetile);

            if (weight_has_value) {  // read a tile from weight tensor
                cb_reserve_back(cb_id_weight, onetile);
                uint32_t l1_weight_write_addr = get_write_ptr(cb_id_weight);
                noc_async_read_tile(tile_offset, weight, l1_weight_write_addr);
                noc_async_read_barrier();
                fill_tile_with_first_element_bfloat16(cb_id_weight);
                cb_push_back(cb_id_weight, onetile);
            }

            if (bias_has_value) {  // read a tile from bias tensor
                cb_reserve_back(cb_id_bias, onetile);
                uint32_t l1_bias_write_addr = get_write_ptr(cb_id_bias);
                noc_async_read_tile(tile_offset, bias, l1_bias_write_addr);
                noc_async_read_barrier();
                fill_tile_with_first_element_bfloat16(cb_id_bias);
                cb_push_back(cb_id_bias, onetile);
            }

            // to read running stats value for updation
            if (is_training_mode) {
                // read a tile from running_mean tensor
                if (running_mean_has_value) {
                    cb_reserve_back(cb_id_running_mean, onetile);
                    uint32_t l1_running_mean_write_addr = get_write_ptr(cb_id_running_mean);
                    noc_async_read_tile(tile_offset, running_mean, l1_running_mean_write_addr);
                    noc_async_read_barrier();
                    fill_tile_with_first_element_bfloat16(cb_id_running_mean);
                    cb_push_back(cb_id_running_mean, onetile);
                }

                // write the updated running_mean
                cb_wait_front(cb_id_updated_running_mean, onetile);
                uint32_t l1_write_updated_running_mean_addr = get_read_ptr(cb_id_updated_running_mean);
                noc_async_write_tile(tile_offset, src, l1_write_updated_running_mean_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_id_updated_running_mean, onetile);
            }

            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // write a tile to dst
                cb_wait_front(cb_id_dst, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_id_dst, onetile);
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
