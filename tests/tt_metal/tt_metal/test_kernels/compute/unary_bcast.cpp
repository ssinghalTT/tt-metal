// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    unary_bcast_init<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_wait_front(tt::CBIndex::c_0, onetile);
    cb_reserve_back(tt::CBIndex::c_16, onetile);

    acquire_dst();

    unary_bcast<BCAST_DIM>(tt::CBIndex::c_0, 0, 0);

    pack_tile(0, tt::CBIndex::c_16);

    release_dst();

    cb_push_back(tt::CBIndex::c_16, onetile);
    cb_pop_front(tt::CBIndex::c_0, onetile);
}
}  // namespace NAMESPACE
