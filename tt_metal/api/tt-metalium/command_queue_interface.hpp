// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <climits>
#include <magic_enum/magic_enum.hpp>
#include <mutex>

#include "cq_commands.hpp"
#include "dispatch_core_manager.hpp"
#include "launch_message_ring_buffer_state.hpp"
#include "memcpy.hpp"
#include "hal.hpp"
#include "dispatch_settings.hpp"
#include "helpers.hpp"

// FIXME: Don't do this in header files
using namespace tt::tt_metal;

namespace tt::tt_metal {

enum class CommandQueueDeviceAddrType : uint8_t {
    PREFETCH_Q_RD = 0,
    // Used to notify host of how far device has gotten, doesn't need L1 alignment because it's only written locally by
    // prefetch kernel.
    PREFETCH_Q_PCIE_RD = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    // Max of 2 CQs. COMPLETION_Q*_LAST_EVENT_PTR track the last completed event in the respective CQs
    COMPLETION_Q0_LAST_EVENT = 4,
    COMPLETION_Q1_LAST_EVENT = 5,
    DISPATCH_S_SYNC_SEM = 6,
    DISPATCH_MESSAGE = 7,
    UNRESERVED = 8
};

enum class CommandQueueHostAddrType : uint8_t {
    ISSUE_Q_RD = 0,
    ISSUE_Q_WR = 1,
    COMPLETION_Q_WR = 2,
    COMPLETION_Q_RD = 3,
    UNRESERVED = 4
};

// Contains constants related to FD
//
// Deprecated note: for constant values, use tt::tt_metal::dispatch::DispatchConstants instead.
//
//
struct [[deprecated]] dispatch_constants {
public:
    dispatch_constants& operator=(const dispatch_constants&) = delete;
    dispatch_constants& operator=(dispatch_constants&& other) noexcept = delete;
    dispatch_constants(const dispatch_constants&) = delete;
    dispatch_constants(dispatch_constants&& other) noexcept = delete;

    static const dispatch_constants& get(const CoreType& core_type, const uint32_t num_hw_cqs = 0) {
        if (num_hw_cqs > 0 && (num_hw_cqs != hw_cqs || core_type != last_core_type || !inst)) {
            hw_cqs = num_hw_cqs;
            last_core_type = core_type;
            inst = std::unique_ptr<dispatch_constants>(new dispatch_constants(core_type, hw_cqs));
        }

        TT_FATAL(hw_cqs > 0, "Command Queue is not initialized.");
        return *inst;
    }

    using prefetch_q_entry_type = uint16_t;

    static constexpr uint8_t MAX_NUM_HW_CQS = 2;
    // Currently arbitrary, can be adjusted as needed at the cost of more L1 memory
    static constexpr uint32_t DISPATCH_MESSAGE_ENTRIES = 16;
    static constexpr uint32_t DISPATCH_MESSAGES_MAX_OFFSET =
        std::numeric_limits<decltype(go_msg_t::dispatch_message_offset)>::max();
    static_assert(
        dispatch_constants::DISPATCH_MESSAGE_ENTRIES <=
        sizeof(decltype(CQDispatchCmd::notify_dispatch_s_go_signal.index_bitmask)) * CHAR_BIT);
    // Currently arbitrary, can be adjusted as needed at the cost of more static memory
    static constexpr uint32_t DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES = 64;
    static constexpr uint32_t GO_SIGNAL_BITS_PER_TXN_TYPE = 4;
    static constexpr uint32_t GO_SIGNAL_MAX_TXNS_PER_TYPE = 1 << GO_SIGNAL_BITS_PER_TXN_TYPE - 1;

    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;
    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
    // dispatch_s CB page size is 128 bytes. This should currently be enough to accomodate all commands that
    // are sent to it. Change as needed, once this endpoint is required to handle more than go signal mcasts.
    static constexpr uint32_t DISPATCH_S_BUFFER_LOG_PAGE_SIZE = 7;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;
    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken
    // down into equal sized partial pages BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is
    // incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    uint32_t prefetch_q_entries() const { return prefetch_q_entries_; }

    uint32_t prefetch_q_size() const { return prefetch_q_size_; }

    uint32_t max_prefetch_command_size() const { return max_prefetch_command_size_; }

    uint32_t cmddat_q_base() const { return cmddat_q_base_; }

    uint32_t cmddat_q_size() const { return cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_base() const { return dispatch_buffer_base_; }

    uint32_t dispatch_buffer_pages() const { return dispatch_buffer_pages_; }

    uint32_t prefetch_d_buffer_size() const { return prefetch_d_buffer_size_; }

    uint32_t prefetch_d_buffer_pages() const { return prefetch_d_buffer_pages_; }

    uint32_t mux_buffer_size(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_size_ / num_hw_cqs; }

    uint32_t mux_buffer_pages(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_pages_ / num_hw_cqs; }

    uint32_t dispatch_s_buffer_size() const { return dispatch_s_buffer_size_; }

    uint32_t dispatch_s_buffer_pages() const {
        return dispatch_s_buffer_size_ / (1 << tt::tt_metal::dispatch::DispatchConstants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    }

    uint32_t get_device_command_queue_addr(const CommandQueueDeviceAddrType& device_addr_type) const {
        uint32_t index = tt::utils::underlying_type<CommandQueueDeviceAddrType>(device_addr_type);
        TT_ASSERT(index < this->device_cq_addrs_.size());
        return device_cq_addrs_[index];
    }

    uint32_t get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const {
        return tt::utils::underlying_type<CommandQueueHostAddrType>(host_addr) *
               tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST);
    }

    uint32_t get_dispatch_message_offset(uint32_t index) const {
        TT_ASSERT(index < tt::tt_metal::dispatch::DispatchConstants::DISPATCH_MESSAGE_ENTRIES);
        uint32_t offset = index * hal.get_alignment(HalMemType::L1);
        return offset;
    }

private:
    dispatch_constants(const CoreType& core_type, const uint32_t num_hw_cqs) {
        using namespace tt::tt_metal::dispatch;

        // TODO: This is hardcoded to use defaults for now
        const auto settings = DispatchSettings::defaults(core_type, tt::Cluster::instance(), num_hw_cqs);
        prefetch_q_entries_ = settings.prefetch_q_entries_;
        max_prefetch_command_size_ = settings.prefetch_max_cmd_size_;
        cmddat_q_size_ = settings.prefetch_cmddat_q_size_;
        scratch_db_size_ = settings.prefetch_scratch_db_size_;
        prefetch_d_buffer_size_ = settings.prefetch_d_buffer_size_;
        dispatch_s_buffer_size_ = settings.dispatch_s_buffer_size_;
        const auto dispatch_buffer_block_size = settings.dispatch_size_;
        const auto [l1_base, l1_size] = get_device_l1_info(core_type);
        const auto pcie_alignment = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST);
        const auto l1_alignment = tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::L1);

        TT_ASSERT(cmddat_q_size_ >= 2 * max_prefetch_command_size_);
        TT_ASSERT(scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
        TT_ASSERT(
            DispatchConstants::DISPATCH_MESSAGE_ENTRIES <= DispatchConstants::DISPATCH_MESSAGES_MAX_OFFSET / l1_alignment + 1,
            "Number of dispatch message entries exceeds max representable offset");

        uint8_t num_dev_cq_addrs = magic_enum::enum_count<CommandQueueDeviceAddrType>();
        std::vector<uint32_t> device_cq_addr_sizes_(num_dev_cq_addrs, 0);
        for (auto dev_addr_idx = 0; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
            CommandQueueDeviceAddrType dev_addr_type =
                magic_enum::enum_cast<CommandQueueDeviceAddrType>(dev_addr_idx).value();
            if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_RD) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_rd_ptr_size_;
            } else if (dev_addr_type == CommandQueueDeviceAddrType::PREFETCH_Q_PCIE_RD) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.prefetch_q_pcie_rd_ptr_size_;
            } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.dispatch_s_sync_sem_;
            } else if (dev_addr_type == CommandQueueDeviceAddrType::DISPATCH_MESSAGE) {
                device_cq_addr_sizes_[dev_addr_idx] = settings.dispatch_message_;
            } else {
                device_cq_addr_sizes_[dev_addr_idx] = settings.other_ptrs_size;
            }
        }

        device_cq_addrs_.resize(num_dev_cq_addrs);
        device_cq_addrs_[0] = l1_base;
        for (auto dev_addr_idx = 1; dev_addr_idx < num_dev_cq_addrs; dev_addr_idx++) {
            device_cq_addrs_[dev_addr_idx] =
                device_cq_addrs_[dev_addr_idx - 1] + device_cq_addr_sizes_[dev_addr_idx - 1];
            CommandQueueDeviceAddrType dev_addr_type = magic_enum::enum_value<CommandQueueDeviceAddrType>(dev_addr_idx);
            if (dev_addr_type == CommandQueueDeviceAddrType::UNRESERVED) {
                device_cq_addrs_[dev_addr_idx] = align_addr(device_cq_addrs_[dev_addr_idx], pcie_alignment);
            }
        }

        prefetch_q_size_ = prefetch_q_entries_ * sizeof(prefetch_q_entry_type);
        uint32_t prefetch_dispatch_unreserved_base =
            device_cq_addrs_[tt::utils::underlying_type<CommandQueueDeviceAddrType>(
                CommandQueueDeviceAddrType::UNRESERVED)];
        cmddat_q_base_ = prefetch_dispatch_unreserved_base + align_size(prefetch_q_size_, pcie_alignment);
        scratch_db_base_ = cmddat_q_base_ + align_size(cmddat_q_size_, pcie_alignment);

        TT_ASSERT(scratch_db_base_ + scratch_db_size_ < l1_size);
        dispatch_buffer_base_ = align_addr(prefetch_dispatch_unreserved_base, 1 << DispatchConstants::DISPATCH_BUFFER_LOG_PAGE_SIZE);
        dispatch_buffer_pages_ = dispatch_buffer_block_size / (1 << DispatchConstants::DISPATCH_BUFFER_LOG_PAGE_SIZE);
        dispatch_buffer_block_size_pages_ = dispatch_buffer_pages_ / settings.dispatch_pages_per_block_;
        const uint32_t dispatch_cb_end = dispatch_buffer_base_ + settings.dispatch_size_;
        TT_ASSERT(dispatch_cb_end < l1_size);
        prefetch_d_buffer_pages_ = settings.prefetch_d_pages_;
    }

    std::pair<uint32_t, uint32_t> get_device_l1_info(const CoreType& core_type) const {
        uint32_t l1_base;
        uint32_t l1_size;
        if (core_type == CoreType::WORKER) {
            l1_base = hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
            l1_size =
                hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
        } else if (core_type == CoreType::ETH) {
            l1_base = hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
            l1_size =
                hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
        } else {
            TT_THROW("get_base_device_command_queue_addr not implemented for core type");
        }

        return {l1_base, l1_size};
    }

    uint32_t prefetch_q_entries_;
    uint32_t prefetch_q_size_;
    uint32_t max_prefetch_command_size_;
    uint32_t cmddat_q_base_;
    uint32_t cmddat_q_size_;
    uint32_t scratch_db_base_;
    uint32_t scratch_db_size_;
    uint32_t dispatch_buffer_base_;
    uint32_t dispatch_buffer_block_size_pages_;
    uint32_t dispatch_buffer_pages_;
    uint32_t prefetch_d_buffer_size_;
    uint32_t prefetch_d_buffer_pages_;
    uint32_t dispatch_s_buffer_size_;
    std::vector<uint32_t> device_cq_addrs_;
    static inline std::unique_ptr<dispatch_constants> inst;
    static inline uint32_t hw_cqs;
    static inline CoreType last_core_type = CoreType::WORKER;
};

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

inline uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    using namespace tt::tt_metal::dispatch;

    return (DispatchConstants::MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) + ((channel >> 2) * DispatchConstants::MAX_DEV_CHANNEL_SIZE) +
           get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::dispatch::DispatchConstants::MAX_DEV_CHANNEL_SIZE;
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t issue_q_rd_ptr =
        dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        issue_q_rd_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t issue_q_wr_ptr =
        dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    tt::Cluster::instance().read_sysmem(
        &recv, sizeof(uint32_t), issue_q_wr_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::dispatch::DispatchConstants::MAX_DEV_CHANNEL_SIZE;
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t completion_q_wr_ptr =
        dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        completion_q_wr_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t completion_q_rd_ptr =
        dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::Cluster::instance().read_sysmem(
        &recv, sizeof(uint32_t), completion_q_rd_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the
    // completion region Equation for issue fifo size is | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start) :
        cq_start(cq_start),
        command_completion_region_size(
            (((cq_size - cq_start) / dispatch_constants::TRANSFER_PAGE_SIZE) / 4) *
            dispatch_constants::TRANSFER_PAGE_SIZE),
        command_issue_region_size((cq_size - cq_start) - this->command_completion_region_size),
        issue_fifo_size(command_issue_region_size >> 4),
        issue_fifo_limit(
            ((cq_start + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
        completion_fifo_size(command_completion_region_size >> 4),
        completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
        offset(get_absolute_cq_offset(channel, cq_id, cq_size)),
        id(cq_id) {
        TT_ASSERT(
            this->command_completion_region_size % hal.get_alignment(HalMemType::HOST) == 0 and
                this->command_issue_region_size % hal.get_alignment(HalMemType::HOST) == 0,
            "Issue queue and completion queue need to be {}B aligned!",
            hal.get_alignment(HalMemType::HOST));
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (this->cq_start + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B
    // aligned and remaining space is dedicated for completion queue Smaller issue queues can lead to more stalls for
    // applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t cq_start;
    const uint32_t command_completion_region_size;
    const uint32_t command_issue_region_size;
    const uint8_t id;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;

    // TODO add the host addresses from dispatch constants in here
};

class SystemMemoryManager {
private:
    chip_id_t device_id;
    uint8_t num_hw_cqs;
    const std::function<void(uint32_t, uint32_t, const uint8_t*)> fast_write_callable;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;
    std::vector<int> cq_to_event;
    std::vector<int> cq_to_last_completed_event;
    std::vector<std::mutex> cq_to_event_locks;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<tt::Writer> prefetch_q_writers;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset;
    std::array<LaunchMessageRingBufferState, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>
        worker_launch_message_buffer_state;

public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
        device_id(device_id),
        num_hw_cqs(num_hw_cqs),
        fast_write_callable(tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
        bypass_enable(false),
        bypass_buffer_write_offset(0) {
        using namespace tt::tt_metal::dispatch;

        this->completion_byte_addrs.resize(num_hw_cqs);
        this->prefetcher_cores.resize(num_hw_cqs);
        this->prefetch_q_writers.reserve(num_hw_cqs);
        this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
        this->prefetch_q_dev_fences.resize(num_hw_cqs);

        // Split hugepage into however many pieces as there are CQs
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        char* hugepage_start = (char*)tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        hugepage_start += (channel >> 2) * DispatchConstants::MAX_DEV_CHANNEL_SIZE;
        this->cq_sysmem_start = hugepage_start;

        // TODO(abhullar): Remove env var and expose sizing at the API level
        char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
        if (cq_size_override_env != nullptr) {
            uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
            this->cq_size = cq_size_override;
        } else {
            this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
            if (tt::Cluster::instance().is_galaxy_cluster()) {
                // We put 4 galaxy devices per huge page since number of hugepages available is less than number of
                // devices.
                this->cq_size = this->cq_size / DispatchConstants::DEVICES_PER_UMD_CHANNEL;
            }
        }
        this->channel_offset = DispatchConstants::MAX_HUGEPAGE_SIZE * get_umd_channel(channel) + (channel >> 2) * DispatchConstants::MAX_DEV_CHANNEL_SIZE;

        CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(device_id);
        uint32_t completion_q_rd_ptr = dispatch_constants::get(core_type).get_device_command_queue_addr(
            CommandQueueDeviceAddrType::COMPLETION_Q_RD);
        uint32_t prefetch_q_base =
            dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        uint32_t cq_start =
            dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair prefetcher_core =
                tt::tt_metal::dispatch_core_manager::instance().prefetcher_core(device_id, channel, cq_id);
            auto prefetcher_virtual = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
            this->prefetcher_cores[cq_id] = tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
            this->prefetch_q_writers.emplace_back(
                tt::Cluster::instance().get_static_tlb_writer(this->prefetcher_cores[cq_id]));

            tt_cxy_pair completion_queue_writer_core =
                tt::tt_metal::dispatch_core_manager::instance().completion_queue_writer_core(device_id, channel, cq_id);
            auto completion_queue_writer_virtual =
                tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
                    completion_queue_writer_core.chip,
                    CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
                    core_type);

            const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data =
                tt::Cluster::instance()
                    .get_tlb_data(tt_cxy_pair(
                        completion_queue_writer_core.chip,
                        completion_queue_writer_virtual.x,
                        completion_queue_writer_virtual.y))
                    .value();
            auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
            this->completion_byte_addrs[cq_id] = completion_tlb_offset + completion_q_rd_ptr % completion_tlb_size;

            this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size, cq_start));
            // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
            // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping Plus
            // 1 to allow us to start writing to issue queue before we reserve space in the prefetch queue
            TT_FATAL(
                dispatch_constants::get(core_type, num_hw_cqs).max_prefetch_command_size() *
                        (dispatch_constants::get(core_type, num_hw_cqs).prefetch_q_entries() + 2) <=
                    this->get_issue_queue_size(cq_id),
                "Issue queue for cq_id {} has size of {} which is too small",
                cq_id,
                this->get_issue_queue_size(cq_id));
            this->cq_to_event.push_back(0);
            this->cq_to_last_completed_event.push_back(0);
            this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
            this->prefetch_q_dev_fences[cq_id] =
                prefetch_q_base + dispatch_constants::get(core_type, num_hw_cqs).prefetch_q_entries() *
                                      sizeof(dispatch_constants::prefetch_q_entry_type);
        }
        std::vector<std::mutex> temp_mutexes(num_hw_cqs);
        cq_to_event_locks.swap(temp_mutexes);
    }

    uint32_t get_next_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t next_event = ++this->cq_to_event[cq_id];  // Event ids start at 1
        cq_to_event_locks[cq_id].unlock();
        return next_event;
    }

    void reset_event_id(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] = 0;
        cq_to_event_locks[cq_id].unlock();
    }

    void increment_event_id(const uint8_t cq_id, const uint32_t val) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] += val;
        cq_to_event_locks[cq_id].unlock();
    }

    void set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
        TT_ASSERT(
            event_id >= this->cq_to_last_completed_event[cq_id],
            "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
            "completed event is {}",
            event_id,
            this->cq_to_last_completed_event[cq_id]);
        cq_to_event_locks[cq_id].lock();
        this->cq_to_last_completed_event[cq_id] = event_id;
        cq_to_event_locks[cq_id].unlock();
    }

    uint32_t get_last_completed_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
        cq_to_event_locks[cq_id].unlock();
        return last_completed_event;
    }

    void reset(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = 0;
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = 0;
    }

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_size = (issue_queue_size >> 4);
        cq_interface.issue_fifo_limit = (cq_interface.cq_start + cq_interface.offset + issue_queue_size) >> 4;
    }

    void set_bypass_mode(const bool enable, const bool clear) {
        this->bypass_enable = enable;
        if (clear) {
            this->bypass_buffer.clear();
            this->bypass_buffer_write_offset = 0;
        }
    }

    bool get_bypass_mode() { return this->bypass_enable; }

    std::vector<uint32_t>& get_bypass_data() { return this->bypass_buffer; }

    uint32_t get_issue_queue_size(const uint8_t cq_id) const { return this->cq_interfaces[cq_id].issue_fifo_size << 4; }

    uint32_t get_issue_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
    }

    uint32_t get_completion_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_size << 4;
    }

    uint32_t get_completion_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
    }

    uint32_t get_issue_queue_write_ptr(const uint8_t cq_id) const {
        if (this->bypass_enable) {
            return this->bypass_buffer_write_offset;
        } else {
            return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
        }
    }

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
    }

    uint32_t get_completion_queue_read_toggle(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
    }

    uint32_t get_cq_size() const { return this->cq_size; }

    chip_id_t get_device_id() const { return this->device_id; }

    std::vector<SystemMemoryCQInterface>& get_cq_interfaces() { return this->cq_interfaces; }

    void* issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) {
            uint32_t curr_size = this->bypass_buffer.size();
            uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
            this->bypass_buffer.resize(new_size);
            return (void*)((char*)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
        }

        uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

        const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
        if (issue_q_write_ptr + tt::tt_metal::dispatch::align_addr(cmd_size_B, tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST)) >
            command_issue_limit) {
            this->wrap_issue_queue_wr_ptr(cq_id);
            issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
        }

        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void* issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

        return issue_q_region;
    }

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        if (this->bypass_enable) {
            std::copy((uint8_t*)data, (uint8_t*)data + size_in_bytes, (uint8_t*)this->bypass_buffer.data() + write_ptr);
        } else {
            memcpy_to_device(user_scratchspace, data, size_in_bytes);
        }
    }

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) {
            this->bypass_buffer_write_offset += push_size_B;
            return;
        }

        // All data needs to be PCIE_ALIGNMENT aligned
        uint32_t push_size_16B =
            tt::tt_metal::dispatch::align_addr(push_size_B, tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST)) >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
        uint32_t issue_q_wr_ptr =
            dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);

        if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
            cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
            cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
        } else {
            cq_interface.issue_fifo_wr_ptr += push_size_16B;
        }

        // Also store this data in hugepages, so if a hang happens we can see what was written by host.
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        tt::Cluster::instance().write_sysmem(
            &cq_interface.issue_fifo_wr_ptr,
            sizeof(uint32_t),
            issue_q_wr_ptr + get_relative_cq_offset(cq_id, this->cq_size),
            mmio_device_id,
            channel);
    }

    uint32_t completion_queue_wait_front(const uint8_t cq_id, volatile bool& exit_condition) const {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and
                 cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition);
        return write_ptr_and_toggle;
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

        uint32_t read_ptr_and_toggle =
            cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle);

        // Also store this data in hugepages in case we hang and can't get it from the device.
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
        uint32_t completion_q_rd_ptr =
            dispatch_constants::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
        tt::Cluster::instance().write_sysmem(
            &read_ptr_and_toggle,
            sizeof(uint32_t),
            completion_q_rd_ptr + get_relative_cq_offset(cq_id, this->cq_size),
            mmio_device_id,
            channel);
    }

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        if (this->bypass_enable) {
            return;
        }
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
        uint32_t data_read_B = num_pages_read * dispatch_constants::TRANSFER_PAGE_SIZE;
        uint32_t data_read_16B = data_read_B >> 4;

        SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_completion_queue_read_ptr(cq_id);
    }

    void fetch_queue_reserve_back(const uint8_t cq_id) {
        if (this->bypass_enable) {
            return;
        }

        CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(device_id);
        const uint32_t prefetch_q_rd_ptr =
            dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);

        // Helper to wait for fetch queue space, if needed
        uint32_t fence;
        auto wait_for_fetch_q_space = [&]() {
            // Loop until space frees up
            while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
                tt::Cluster::instance().read_core(
                    &fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
                this->prefetch_q_dev_fences[cq_id] = fence;
            }
        };

        wait_for_fetch_q_space();

        // Wrap FetchQ if possible
        uint32_t prefetch_q_base =
            dispatch_constants::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
        uint32_t prefetch_q_limit =
            prefetch_q_base + dispatch_constants::get(core_type, num_hw_cqs).prefetch_q_entries() *
                                  sizeof(dispatch_constants::prefetch_q_entry_type);
        if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
            this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
            wait_for_fetch_q_space();
        }
    }

    void fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher = false) {
        CoreType dispatch_core_type =
            tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
        uint32_t max_command_size_B =
            dispatch_constants::get(dispatch_core_type, num_hw_cqs).max_prefetch_command_size();
        TT_ASSERT(
            command_size_B <= max_command_size_B,
            "Generated prefetcher command of size {} B exceeds max command size {} B",
            command_size_B,
            max_command_size_B);
        TT_ASSERT(
            (command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF,
            "FetchQ command too large to represent");
        if (this->bypass_enable) {
            return;
        }
        tt_driver_atomics::sfence();
        dispatch_constants::prefetch_q_entry_type command_size_16B =
            command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE;

        // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
        // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
        // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
        // prefetcher is MSB of FetchQ entry.
        if (stall_prefetcher) {
            command_size_16B |= (1 << ((sizeof(dispatch_constants::prefetch_q_entry_type) * 8) - 1));
        }
        this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
        this->prefetch_q_dev_ptrs[cq_id] += sizeof(dispatch_constants::prefetch_q_entry_type);
    }

    std::array<LaunchMessageRingBufferState, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>&
    get_worker_launch_message_buffer_state() {
        return this->worker_launch_message_buffer_state;
    }

    void reset_worker_launch_message_buffer_state(const uint32_t num_entries) {
        std::for_each(
            this->worker_launch_message_buffer_state.begin(),
            this->worker_launch_message_buffer_state.begin() + num_entries,
            std::mem_fn(&LaunchMessageRingBufferState::reset));
    }
};

}  // namespace tt::tt_metal
