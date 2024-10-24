// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>

void kernel_launch(uint32_t kernel_base_addr) {

#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __fw_export_end_text[];
    uint32_t x = kernel_base_addr + (uint32_t) __kernel_init_local_l1_base - (uint32_t)__fw_export_end_text;
    firmware_kernel_common_init((void tt_l1_ptr *)x);
    extern uint32_t __kernel_data_lma[];
    static uint32_t u [[gnu::used]] = (uint32_t)&__kernel_data_lma;
    static uint32_t v [[gnu::used]] = x;

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    } else {
        noc_local_state_init(NOC_0);
        noc_local_state_init(NOC_1);
    }

    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        kernel_main();
    }
#endif
}
