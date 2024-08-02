// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "dev_msgs.h"
#include "stream_io_map.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"

#include "debug/status.h"
#include "debug/dprint.h"
// clang-format on

uint32_t halt_stack_ptr_save;

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t *const ncrisc_run = &mailboxes->slave_sync.ncrisc;

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t atomic_ret_val __attribute__((section("l1_data"))) __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr *rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr *crta_l1_base __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
    uint16_t core_flat_id __attribute__((used));
    uint32_t nocWriteSize __attribute__((used));
    uint32_t *nocWriteBuffer __attribute__((used));
    uint32_t *nocWriteIndex __attribute__((used));
}
#endif

extern "C" void ncrisc_resume(void);
extern "C" void notify_brisc_and_halt(uint32_t status);

inline __attribute__((always_inline)) void set_ncrisc_resume_addr() {
#ifdef NCRISC_HAS_IRAM
    mailboxes->ncrisc_halt.resume_addr = (uint32_t)ncrisc_resume;
#endif
}

inline __attribute__((always_inline)) void notify_brisc_and_wait() {
#ifdef NCRISC_HAS_IRAM
    notify_brisc_and_halt(RUN_SYNC_MSG_DONE);
#else
    while (*ncrisc_run != RUN_SYNC_MSG_GO);
#endif
}

inline __attribute__((always_inline)) void signal_ncrisc_completion() {
#ifndef NCRISC_HAS_IRAM
    *ncrisc_run = RUN_SYNC_MSG_DONE;
#endif
}

int main(int argc, char *argv[]) {
    DEBUG_STATUS("I");

    disable_lowcache();

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint *)__ldm_data_start, (uint tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE, num_words);

    risc_init();

    // If NCRISC has IRAM it needs to halt before BRISC copies data from L1 to IRAM
    // Need to save address to jump to after BRISC resumes NCRISC
    set_ncrisc_resume_addr();

    // Cleanup profiler buffer incase we never get the go message
    while (1) {
        // DEBUG_STATUS("W");
        notify_brisc_and_wait();
        DeviceZoneScopedMainN("NCRISC-FW");

        setup_cb_read_write_interfaces(0, mailboxes->launch.kernel_config.max_cb_index, true, true);

        uint32_t kernel_config_base = mailboxes->launch.kernel_config.kernel_config_base;
        rta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_TENSIX_DM1].rta_offset);
        crta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base +
            mailboxes->launch.kernel_config.mem_map[DISPATCH_CLASS_TENSIX_DM1].crta_offset);

        // DEBUG_STATUS("R");
        kernel_init();
        DEBUG_STATUS("D");

        signal_ncrisc_completion();
    }

    return 0;
}
