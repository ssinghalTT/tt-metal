// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "debug/dprint.h"

#include "dataflow_api.h"
#include "utils/bfloat16.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_intermed0 = get_compile_time_arg_val(1);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr uint32_t in0_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t C = get_compile_time_arg_val(7);
    constexpr uint32_t H = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);
    constexpr uint32_t dim = get_compile_time_arg_val(10);
    constexpr uint32_t all = get_compile_time_arg_val(11);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = in0_stick_size};

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};

    // Use cb as L1 scratch memory
    uint32_t out_addr = get_write_ptr(cb_id_intermed0);
    volatile tt_l1_ptr uint32_t* max_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    uint32_t max_index = 0;
    uint32_t max_val = 0;
    uint32_t index_counter = 0;

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t l = 0; l < W; l++) {
                    uint32_t max_index = 0;
                    uint32_t index_counter = 0;
                    max_val = input_ptr[0];

                    noc_async_read_page(b * C * H * W + c * H * W + h * W, s0, cb_addr);
                    noc_async_read_barrier();
                    uint16_t val = input_ptr[b * C * H * W + c * H * W + h * W + l];
                    DPRINT << b << "x" << c << "x" << h << "x" << l << " : idx "
                           << b * C * H * W + c * H * W + h * W + l << " val " << val << ENDL();

                    if (bfloat16_greater(val, max_val)) {
                        max_index = h;
                        max_val = val;
                    }

                    // Store the index of the max value for the current (b, c, l)
                    max_vals[b * C * W + c * W + l] = max_index;
                }
            }
        }
    }

    if (dim == 0) {
        for (uint32_t k = 0; k < C; k++) {
            for (uint32_t j = 0; j < H; j++) {
                for (uint32_t l = 0; l < W; l++) {
                    uint32_t max_index = 0;
                    uint16_t max_val = input_ptr[0];
                    uint32_t index_counter = 0;

                    for (uint32_t i = 0; i < B; i++) {
                        noc_async_read_page(i * C * H * W + k * H * W + j * W + l, s0, cb_addr);
                        noc_async_read_barrier();
                        uint16_t val = input_ptr[i * C * H * W + k * H * W + j * W + l];

                        if (bfloat16_greater(val, max_val)) {
                            max_index = i;
                            max_val = val;
                        }
                    }

                    max_vals[k * H * W + j * W + l] = max_index;
                }
            }
        }
    } else if (dim == 1) {
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t j = 0; j < H; j++) {
                for (uint32_t l = 0; l < W; l++) {
                    uint32_t max_index = 0;
                    uint32_t index_counter = 0;
                    max_val = input_ptr[0];
                    for (uint32_t c = 0; c < C; c++) {
                        noc_async_read_page(b * C * H * W + c * H * W + j * W + l, s0, cb_addr);
                        noc_async_read_barrier();
                        uint16_t val = input_ptr[b * C * H * W + c * H * W + j * W + l];

                        if (bfloat16_greater(val, max_val)) {
                            max_index = c;
                            max_val = val;
                        }
                    }

                    max_vals[b * H * W + j * W + l] = max_index;
                }
            }
        }
    } else if (dim == 2) {
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t c = 0; c < C; c++) {
                for (uint32_t l = 0; l < W; l++) {
                    uint32_t max_index = 0;
                    uint32_t index_counter = 0;
                    max_val = input_ptr[0];
                    for (uint32_t h = 0; h < H; h++) {
                        noc_async_read_page(b * C * H * W + c * H * W + h * W + l, s0, cb_addr);
                        noc_async_read_barrier();
                        uint16_t val = input_ptr[b * C * H * W + c * H * W + h * W + l];
                        // DPRINT<<b<<"x"<<c<<"x"<<h<<"x"<<l<<" : idx "<< b * C * H * W + c * H * W + h * W + l << " val
                        // " << val << ENDL();

                        if (bfloat16_greater(val, max_val)) {
                            max_index = h;
                            max_val = val;
                        }
                    }

                    // Store the index of the max value for the current (b, c, l)
                    max_vals[b * C * W + c * W + l] = max_index;
                }
            }
        }
    } else if (dim == 3) {
        for (uint32_t l = 0; l < B; l++) {
            for (uint32_t k = 0; k < C; k++) {
                for (uint32_t j = 0; j < H; j++) {
                    noc_async_read_page(l * C * H + k * H + j, s0, cb_addr);
                    noc_async_read_barrier();
                    index_counter = 0;
                    max_index = 0;
                    max_val = input_ptr[0];
                    for (uint32_t i = 0; i < W; i++) {
                        uint16_t val = input_ptr[i];
                        if (bfloat16_greater(val, max_val)) {
                            max_index = index_counter;
                            max_val = val;
                        }
                        index_counter++;
                    }
                    max_vals[l * C * H + k * H + j] = max_index;
                }
            }
        }
    }
    // TODO: Generalize write for argmax for other dims
    if constexpr (all) {
        max_vals[0] = max_index;
    }
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);

    noc_async_write(out_addr, dst_noc_addr, out_stick_size);
    noc_async_write_barrier();
}
