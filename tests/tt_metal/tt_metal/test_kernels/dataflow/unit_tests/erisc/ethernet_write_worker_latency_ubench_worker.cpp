// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "debug/debug.h"

static constexpr uint32_t num_iters = get_compile_time_arg_val(0);
static constexpr uint32_t read_address = get_compile_time_arg_val(1);

void kernel_main() {
    tt_l1_ptr uint8_t* ptr = reinterpret_cast<tt_l1_ptr uint8_t*>(read_address);

    for (uint32_t i = 0; i < num_iters; ++i) {
        DPRINT << ptr[i] << ENDL();
    }
}
