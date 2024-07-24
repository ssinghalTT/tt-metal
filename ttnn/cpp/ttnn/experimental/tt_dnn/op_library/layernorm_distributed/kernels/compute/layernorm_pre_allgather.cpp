// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes larnorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and E(x) in the left most columns per tile.
 * For rmsnorm it computes E(x**2) and returns it as a one tile wide output tensor containing E(x**2) in the left most column per tile.
*/

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"


ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }


namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_reduce = tt::CB::c_in1;

    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t cb_x2 = tt::CB::c_intermed0; // x**2

    cb_wait_front(cb_reduce, 1); // comes from the reader

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    mm_init(cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {

        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * x**2
         */
        unpack_reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_inp, wt+blk); // cumulative wait
            cb_reserve_back(cb_x2, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr<blk; wtr++) {
                mul_tiles(cb_inp, cb_inp, wt+wtr, wt+wtr, wtr);
                pack_tile(wtr, cb_x2, wt+wtr);
            }
            REL();
            cb_push_back(cb_x2, blk);
        }

        /*
         * sum(x**2)
         */
        unpack_reconfig_data_format(cb_x2, cb_reduce);
        pack_reconfig_data_format(cb_out);
        mm_init_short(cb_reduce, cb_x2, 1);
        cb_wait_front(cb_x2, Wt);
        cb_reserve_back(cb_out, onetile);
        ACQ();
        for (uint32_t wtr = 0; wtr<Wt; wtr++) {
            matmul_tiles(cb_x2, cb_reduce, wtr, 0, dst0, false);
        }
        pack_tile(dst0, cb_out, 0);
        REL();
        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_x2, Wt);


        #ifndef RMSNORM

        /*
         * sum(x)
         */
        unpack_reconfig_data_format(cb_inp, cb_reduce);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, onetile);
        ACQ();
        for (uint32_t wtr = 0; wtr<Wt; wtr++) {
            matmul_tiles(cb_inp, cb_reduce, wtr, 0, dst0, false);
        }
        pack_tile(dst0, cb_out, 1);
        REL();
        cb_push_back(cb_out, onetile);

        #endif

        cb_pop_front(cb_inp, Wt);
    }
    cb_pop_front(cb_reduce, 1);
}
}
