// SPDX-FileCopyrightText: © 2024 Martin Chang
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Now we loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First we make sure there is space in the circular buffers
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);  // wait until we have 1 free slot. This blocks if the
                                     // other kernels cannot consume the tiles fast enough.
                                     // Deciding how large the buffer should be is a tradeoff.
        noc_async_read_tile<true>(i, cb_in0, a_addr);  // read the tile into the circular buffer
        noc_async_read_tile<true>(i, cb_in1, b_addr);  // We can overlap async reads and writes
                                                       // to reduce the data movement overhead.

        // NOTE: Since circular buffers are backed by SRAM, we can actually access them by
        // casting the address to a pointer. This is not helpful in most cases as the CPU
        // is quite slow compared to the tensor/simd engines. But useful for debugging.
        // uint16_t* ptr = (uint16_t*)cb_in0_addr;
        // DPRINT << "cb_in0_addr: " << ptr << " " << *ptr;

        noc_async_read_barrier();  // Wait until tile reads are done
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
    }
}
