// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "core_config.h"
#include "dev_mem_map.h"
#include <dev_msgs.h>
#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"
#include "tensix.h"

#include "hal.hpp"

#include "hal_asserts.hpp"

// FIXME: Eventually this file will be gone
#include "hostdevcommon/common_runtime_address_map.h"  // L1_KERNEL_CONFIG_BASE

#include "umd/device/tt_soc_descriptor.h"  // CoreType

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t*)MEM_MAILBOX_BASE)->x))

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been
// committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE =
    ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

static constexpr float EPS_GS = 0.001953125f;
static constexpr float NAN_GS = 6.9752e19;
static constexpr float INF_GS = 1.6948e38;

namespace tt {

namespace tt_metal {

void Hal::initialize_gs() {
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));

    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        ((L1_KERNEL_CONFIG_BASE + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_LOCAL_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_LOCAL_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SCRATCH;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_TRISC_LOCAL_SIZE; // TRISC, BRISC, or NCRISC?
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SIZE;

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumTensixDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types;
    for (uint8_t processor_class_idx = 0; processor_class_idx < NumTensixDispatchClasses; processor_class_idx++) {
        uint32_t num_processors = processor_class_idx == (NumTensixDispatchClasses - 1) ? 3 : 1;
        processor_types.resize(num_processors);
        for (size_t processor_type_idx = 0; processor_type_idx < processor_types.size(); processor_type_idx++) {
            DeviceAddr fw_base, local_init, fw_launch;
            uint32_t fw_launch_value;
            switch (processor_class_idx) {
                case 0: {
                    fw_base = MEM_BRISC_FIRMWARE_BASE;
                    local_init = MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                    fw_launch = 0x0; // BRISC is hardcoded to have reset PC of 0
                    fw_launch_value = generate_risc_startup_addr(fw_base);
                }
                break;
                case 1: {
                    fw_base = MEM_NCRISC_FIRMWARE_BASE;
                    local_init = MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                    fw_launch = 0; // fix me
                    fw_launch_value = fw_base;
                }
                break;
                case 2: {
                    switch (processor_type_idx) {
                        case 0: {
                            fw_base = MEM_TRISC0_FIRMWARE_BASE;
                            local_init = MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = 0; // fix me
                            fw_launch_value = fw_base;
                        }
                        break;
                        case 1: {
                            fw_base = MEM_TRISC1_FIRMWARE_BASE;
                            local_init = MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = 0; // fix me
                            fw_launch_value = fw_base;
                        }
                        break;
                        case 2: {
                            fw_base = MEM_TRISC2_FIRMWARE_BASE;
                            local_init = MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = 0; // fix me
                            fw_launch_value = fw_base;
                        }
                        break;
                    }
                } break;
                default: TT_THROW("Unexpected processor class {} for Blackhole Tensix", processor_class_idx);
            }

            processor_types[processor_type_idx] = HalJitBuildConfig{
                .fw_base_addr = fw_base,
                .local_init_addr = local_init,
                .fw_launch_addr = fw_launch,
                .fw_launch_addr_value = fw_launch_value
            };
        }
        processor_classes[processor_class_idx] = processor_types;
    }

    this->core_info_.push_back(
        {HalProgrammableCoreType::TENSIX, CoreType::WORKER, processor_classes, mem_map_bases, mem_map_sizes, true});

    this->dram_bases_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_sizes_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_SIZE;

    this->mem_alignments_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::HOST)] = PCIE_ALIGNMENT;

    this->relocate_func_ = [](uint64_t addr, uint64_t local_init_addr) {
        if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            // Move addresses in the local memory range to l1 (copied by kernel)
            return (addr & ~MEM_LOCAL_BASE) + local_init_addr;
        } else if ((addr & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
            // Move addresses in the NCRISC memory range to l1 (copied by kernel)
            return (addr & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
        }

        // No relocation needed
        return addr;
    };

    this->valid_reg_addr_func_ = [](uint32_t addr) {
        return (
            ((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            ((addr >= NOC0_REGS_START_ADDR) && (addr < NOC0_REGS_START_ADDR + 0x1000)) ||
            ((addr >= NOC1_REGS_START_ADDR) && (addr < NOC1_REGS_START_ADDR + 0x1000)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0));
    };

    this->noc_xy_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_ENCODING(x, y); };
    this->noc_multicast_encoding_func_ = [](uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) {
        return NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end);
    };
    this->noc_mcast_addr_start_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_X(addr); };
    this->noc_mcast_addr_start_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_Y(addr); };
    this->noc_mcast_addr_end_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_X(addr); };
    this->noc_mcast_addr_end_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_Y(addr); };
    this->noc_ucast_addr_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_X(addr); };
    this->noc_ucast_addr_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_Y(addr); };
    this->noc_local_addr_func_ = [](uint64_t addr) -> uint64_t { return NOC_LOCAL_ADDR(addr); };

    this->stack_size_func_ = [](uint32_t type) -> uint32_t {
        switch (type) {
            case DebugBrisc: return MEM_BRISC_STACK_SIZE;
            case DebugNCrisc: return MEM_NCRISC_STACK_SIZE;
            case DebugErisc: return 0;  // Not managed/checked by us.
            case DebugIErisc: return MEM_IERISC_STACK_SIZE;
            case DebugSlaveIErisc: return MEM_BRISC_STACK_SIZE;
            case DebugTrisc0: return MEM_TRISC0_STACK_SIZE;
            case DebugTrisc1: return MEM_TRISC1_STACK_SIZE;
            case DebugTrisc2: return MEM_TRISC2_STACK_SIZE;
        }
        return 0xdeadbeef;
    };

    this->num_nocs_ = NUM_NOCS;
    this->noc_addr_node_id_bits_ = NOC_ADDR_NODE_ID_BITS;
    this->noc_coord_reg_offset_ = NOC_COORD_REG_OFFSET;
    this->noc_overlay_start_addr_ = NOC_OVERLAY_START_ADDR;
    this->noc_stream_reg_space_size_ = NOC_STREAM_REG_SPACE_SIZE;
    this->noc_stream_remote_dest_buf_size_reg_index_ = STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX;
    this->noc_stream_remote_dest_buf_start_reg_index_ = STREAM_REMOTE_DEST_BUF_START_REG_INDEX;
    this->coordinate_virtualization_enabled_ = COORDINATE_VIRTUALIZATION_ENABLED;
    this->virtual_worker_start_x_ = VIRTUAL_TENSIX_START_X;
    this->virtual_worker_start_y_ = VIRTUAL_TENSIX_START_Y;

    this->eps_ = EPS_GS;
    this->nan_ = NAN_GS;
    this->inf_ = INF_GS;
}

}  // namespace tt_metal
}  // namespace tt
