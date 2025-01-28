// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint_tensix.h"
#include "debug/dprint.h"
#include "compute_kernel_api/reduce.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    uint32_t row_chunk = get_compile_time_arg_val(3);

    reduce_init<true>(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CB::c_out0);
    cb_wait_front(tt::CBIndex::c_2, 1);  // scaler tile from the reader

    constexpr int onetile = 1;

    // tiles are expected to come in the N C W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reduction will be intertwined
    // H W_chunk represent tiles of the chunk in row major order
    // each column in the chunk will have its intermediate result in a separate tile of DST
    // chunk size is calculated based on the number of available tiles in DST
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        tile order (H, W):
    //        1. chunk: (0, 0); (0, 1); (1, 0); (1, 1); (2, 0); (2, 1);
    //        2. chunk: (0, 2); (0, 3); (1, 2); (1, 3); (2, 2); (2, 3);
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; wt += row_chunk) {
            uint32_t chunk_end = std::min(wt + row_chunk, Wt);
            int reduce_dst_idx = 0;

            // reduction for one chunk
            // accumulation of Ht results in separate DST indexes
            acquire_dst();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                for (uint32_t i = wt; i < chunk_end; ++i) {
                    cb_wait_front(tt::CB::c_in0, onetile);
                    reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
                    // dprint_tensix_dest_reg(0);
                    cb_pop_front(tt::CB::c_in0, onetile);
                    ++reduce_dst_idx;
                }
            }
            for (uint32_t i = wt; i < chunk_end; ++i) {
                cb_reserve_back(tt::CB::c_out0, onetile);
                pack_tile((i - wt), tt::CB::c_out0);
                PACK((print_full_tile(tt::CB::c_out0, 0, true)));
                cb_push_back(tt::CB::c_out0, onetile);
            }
            release_dst();
        }
    }
}
}  // namespace NAMESPACE
