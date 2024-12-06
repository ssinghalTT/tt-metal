// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

template <uint32_t bank_base_address, uint32_t page_size, bool use_vc>
FORCE_INLINE void noc_async_read_tile_dram_sharded(
    uint32_t src_addr, uint32_t dest_addr, uint32_t bank_id = 0, const uint32_t vc = 0) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = src_addr + bank_base_address;
    src_addr_ += bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];

    WAYPOINT("NRTW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_index, get_noc_addr_helper(src_noc_xy, src_addr_), dest_addr, page_size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    WAYPOINT("NRTD");

    if constexpr (use_vc) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr_);           // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);              // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;
}

void kernel_main() {
    constexpr uint32_t tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_loops = get_compile_time_arg_val(2);
    constexpr uint32_t start_x = get_compile_time_arg_val(3);
    constexpr uint32_t start_y = get_compile_time_arg_val(4);
    constexpr uint32_t end_x = get_compile_time_arg_val(5);
    constexpr uint32_t end_y = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);

    constexpr uint32_t block_size_bytes = block_num_tiles * tile_size;

    constexpr uint32_t cb_id_in = 1;  // Sharded cb

    const uint64_t noc_addr = get_noc_multicast_addr(start_x, start_y, end_x, end_y, 0);

    uint32_t local_read_addr = get_read_ptr(cb_id_in);

    DPRINT << "start mcast" << ENDL();

    for (uint32_t i = 0; i < num_loops; ++i) {
        if (i % 10000 == 0) {
            DPRINT << "mcast loop: " << i << ENDL();
        }
        for (uint32_t block = 0; block < num_blocks; ++block) {
            uint64_t multicast_data_addr = noc_addr | local_read_addr;

            noc_async_write_multicast_loopback_src(local_read_addr, multicast_data_addr, block_size_bytes, num_cores);
        }
    }

    noc_async_write_barrier();

    DPRINT << "done mcast" << ENDL();
}
