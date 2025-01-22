// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device_command.hpp>
#include <device.hpp>
#include "dispatch.hpp"

namespace tt::tt_metal {
namespace buffer_dispatch {

// ====== Utility Functions for Writes ======

// Dispatch constants required for writing buffer data
struct BufferDispatchConstants {
    uint32_t issue_queue_cmd_limit;
    uint32_t max_prefetch_cmd_size;
    uint32_t max_data_sizeB;
};

// Dispatch parameters computed during runtime. These are used
// to assemble dispatch commands and compute src + dst offsets
// required to write buffer data.
struct BufferWriteDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t address;
    uint32_t dst_page_index;
    uint32_t page_size_to_write;
    uint32_t total_pages_to_write;
    uint32_t pages_per_txn;
    bool issue_wait;
    IDevice* device;
    uint32_t cq_id;
};

// Parameters specific to interleaved buffers
struct InterleavedBufferWriteDispatchParams : BufferWriteDispatchParams {
    uint32_t write_partial_pages;
    uint32_t padded_buffer_size;
    uint32_t max_num_pages_to_write;
    uint32_t initial_src_addr_offset;
};

// Parameters specific to sharded buffers
struct ShardedBufferWriteDispatchParams : BufferWriteDispatchParams {
    bool width_split;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    uint32_t max_pages_per_shard;
    CoreCoord core;
};

// Generate dispatch constants
BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType dispatch_core_type, uint32_t cq_id) {
    BufferDispatchConstants buf_dispatch_constants;

    buf_dispatch_constants.issue_queue_cmd_limit = sysmem_manager.get_issue_queue_limit(cq_id);
    buf_dispatch_constants.max_prefetch_cmd_size =
        dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    buf_dispatch_constants.max_data_sizeB = buf_dispatch_constants.max_prefetch_cmd_size -
                                            (hal.get_alignment(HalMemType::HOST) * 2);  // * 2 to account for issue

    return buf_dispatch_constants;
}

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferWriteDispatchParams initialize_sharded_buf_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferDispatchConstants& buf_dispatch_constants) {
    ShardedBufferWriteDispatchParams dispatch_params;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.total_pages_to_write = buffer.num_pages();
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.page_size_to_write = buffer.aligned_page_size();
    dispatch_params.dst_page_index = 0;
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;

    TT_FATAL(
        buf_dispatch_constants.max_data_sizeB >= dispatch_params.page_size_to_write,
        "Writing padded page size > {} is currently unsupported for sharded tensors.",
        buf_dispatch_constants.max_data_sizeB);
    return dispatch_params;
}

InterleavedBufferWriteDispatchParams initialize_interleaved_buf_dispatch_params(
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    if (buffer.is_valid_partial_region(region)) {
        TT_FATAL(
            region.offset % buffer.page_size() == 0,
            "Offset {} must be divisible by the buffer page size {}.",
            region.offset,
            buffer.page_size());
        TT_FATAL(
            region.size % buffer.page_size() == 0,
            "Size {} must be divisible by the buffer page size {}.",
            region.size,
            buffer.page_size());
        TT_FATAL(
            (region.size + region.offset) <= buffer.size(),
            "(Size + offset) {} must be <= the buffer size {}.",
            region.size + region.offset,
            buffer.size());
    }

    InterleavedBufferWriteDispatchParams dispatch_params;
    dispatch_params.dst_page_index = region.offset / buffer.page_size();
    uint32_t num_pages = region.size / buffer.page_size();

    uint32_t padded_page_size = buffer.aligned_page_size();
    dispatch_params.total_pages_to_write = num_pages;
    dispatch_params.write_partial_pages = padded_page_size > buf_dispatch_constants.max_data_sizeB;
    dispatch_params.page_size_to_write = padded_page_size;
    dispatch_params.padded_buffer_size = num_pages * padded_page_size;

    if (dispatch_params.write_partial_pages) {
        TT_FATAL(num_pages == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
        uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        while (dispatch_params.padded_buffer_size % partial_size != 0) {
            partial_size += pcie_alignment;
        }
        dispatch_params.page_size_to_write = partial_size;
        dispatch_params.total_pages_to_write = dispatch_params.padded_buffer_size / dispatch_params.page_size_to_write;
    }
    const uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());
    const uint32_t num_pages_round_robined = num_pages / num_banks;
    const uint32_t num_banks_with_residual_pages = num_pages % num_banks;
    const uint32_t num_partial_pages_per_page = padded_page_size / dispatch_params.page_size_to_write;
    const uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

    dispatch_params.max_num_pages_to_write =
        (dispatch_params.write_partial_pages)
            ? (num_pages_round_robined > 0 ? (num_banks * num_partials_round_robined) : num_banks_with_residual_pages)
            : dispatch_params.total_pages_to_write;
    dispatch_params.address = buffer.address();
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

// Populate/Assemble dispatch commands for writing buffer data
void populate_interleaved_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    InterleavedBufferWriteDispatchParams& dispatch_params) {
    uint8_t is_dram = uint8_t(buffer.is_dram());
    TT_ASSERT(
        dispatch_params.dst_page_index <= 0xFFFF,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(dispatch_params.dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch,
        is_dram,
        start_page,
        dispatch_params.address,
        dispatch_params.page_size_to_write,
        dispatch_params.pages_per_txn);

    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    uint32_t full_page_size = buffer.aligned_page_size();  // dispatch_params.page_size_to_write could be a partial
                                                           // page if buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = dispatch_params.page_size_to_write < full_page_size;
    uint32_t buffer_addr_offset = dispatch_params.address - buffer.address();
    const uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - buffer.page_size();
        uint32_t src_address_offset = dispatch_params.initial_src_addr_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
             sysmem_address_offset += dispatch_params.page_size_to_write) {
            uint32_t page_size_to_copy = dispatch_params.page_size_to_write;
            if (src_address_offset + dispatch_params.page_size_to_write > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            command_sequence.add_data(
                (char*)src + src_address_offset, page_size_to_copy, dispatch_params.page_size_to_write);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset =
            (((buffer_addr_offset / dispatch_params.page_size_to_write) * num_banks) + dispatch_params.dst_page_index) *
            buffer.page_size();
        if (buffer.page_size() % buffer.alignment() != 0 and buffer.page_size() != buffer.size()) {
            // If page size is not aligned, we cannot do a contiguous write
            uint32_t src_address_offset = dispatch_params.initial_src_addr_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += dispatch_params.page_size_to_write) {
                command_sequence.add_data(
                    (char*)src + src_address_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                src_address_offset += buffer.page_size();
            }
        } else {
            command_sequence.add_data(
                (char*)src + dispatch_params.initial_src_addr_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void populate_sharded_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params) {
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    auto noc_index = dispatch_downstream_noc;
    const CoreCoord virtual_core =
        buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
    command_sequence.add_dispatch_write_linear(
        0,
        buffer.device()->get_noc_unicast_encoding(noc_index, virtual_core),
        dispatch_params.address,
        data_size_bytes);

    if (dispatch_params.buffer_page_mapping) {
        const auto& page_mapping = *(dispatch_params.buffer_page_mapping);
        uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t*)command_sequence.data();
        for (uint32_t dev_page = dispatch_params.dst_page_index;
             dev_page < dispatch_params.dst_page_index + dispatch_params.pages_per_txn;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset, (char*)(src) + host_page.value() * buffer.page_size(), buffer.page_size());
            }
            dst_offset += dispatch_params.page_size_to_write;
        }
    } else {
        if (buffer.page_size() != dispatch_params.page_size_to_write and buffer.page_size() != buffer.size()) {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            for (uint32_t i = 0; i < dispatch_params.pages_per_txn; ++i) {
                command_sequence.add_data(
                    (char*)src + unpadded_src_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                unpadded_src_offset += buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            command_sequence.add_data((char*)src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

// Issue dispatch commands for writing buffer data
template <typename T>
void issue_buffer_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    T& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    uint32_t num_worker_counters = sub_device_ids.size();
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) +      // CQ_PREFETCH_CMD_RELAY_INLINE
            sizeof(CQDispatchCmd) +  // CQ_DISPATCH_CMD_WRITE_PAGED or CQ_DISPATCH_CMD_WRITE_LINEAR
            data_size_bytes,
        pcie_alignment);
    if (dispatch_params.issue_wait) {
        cmd_sequence_sizeB += hal.get_alignment(HalMemType::HOST) *
                              num_worker_counters;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (dispatch_params.issue_wait) {
        uint32_t dispatch_message_base_addr =
            dispatch_constants::get(dispatch_core_type)
                .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = sub_device_id.to_index();
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr +
                dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
        }
    }
    if constexpr (std::is_same_v<T, ShardedBufferWriteDispatchParams>) {
        populate_sharded_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    } else {
        populate_interleaved_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    }

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level helper functions to write buffer data
void write_interleaved_buffer_to_device(
    const void* src,
    InterleavedBufferWriteDispatchParams& dispatch_params,
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t data_offsetB = hal.get_alignment(HalMemType::HOST);  // data appended after CQ_PREFETCH_CMD_RELAY_INLINE
                                                                  // + CQ_DISPATCH_CMD_WRITE_PAGED
    const uint32_t orig_dst_page_index = dispatch_params.dst_page_index;
    uint32_t total_num_pages_written = 0;
    while (dispatch_params.total_pages_to_write > 0) {
        dispatch_params.issue_wait =
            (dispatch_params.dst_page_index == orig_dst_page_index and
             dispatch_params.address == buffer.address());  // only stall for the first write of the buffer
        if (dispatch_params.issue_wait) {
            data_offsetB *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        }

        uint32_t space_availableB = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(dispatch_params.page_size_to_write);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        dispatch_params.pages_per_txn = std::min(
            std::min((uint32_t)num_pages_available, dispatch_params.max_num_pages_to_write),
            dispatch_params.total_pages_to_write);

        // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
        // To handle larger page offsets move bank base address up and update page offset to be relative to the new
        // bank address
        if (dispatch_params.dst_page_index > 0xFFFF or
            (dispatch_params.pages_per_txn == dispatch_params.max_num_pages_to_write and
             dispatch_params.write_partial_pages)) {
            uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());
            uint32_t num_banks_to_use =
                dispatch_params.write_partial_pages ? dispatch_params.max_num_pages_to_write : num_banks;
            uint32_t residual = dispatch_params.dst_page_index % num_banks_to_use;
            uint32_t num_pages_written_per_bank = dispatch_params.dst_page_index / num_banks_to_use;
            dispatch_params.address += num_pages_written_per_bank * dispatch_params.page_size_to_write;
            dispatch_params.dst_page_index = residual;
        }
        dispatch_params.initial_src_addr_offset = dispatch_params.write_partial_pages
                                                      ? dispatch_params.address - buffer.address()
                                                      : total_num_pages_written * buffer.page_size();

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", dispatch_params.cq_id);

        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        total_num_pages_written += dispatch_params.pages_per_txn;
        dispatch_params.total_pages_to_write -= dispatch_params.pages_per_txn;
        dispatch_params.dst_page_index += dispatch_params.pages_per_txn;
    }
}

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    bool width_split, const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping, Buffer& buffer) {
    return width_split ? buffer_page_mapping->all_cores_
                       : corerange_to_cores(
                             buffer.shard_spec().grid(),
                             buffer.num_cores(),
                             buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
}

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    // Skip writing the padded pages along the bottom
    // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
    // Alternative write each page row into separate commands, or have a strided linear write
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_pages;
    if (dispatch_params.width_split) {
        num_pages = dispatch_params.buffer_page_mapping->core_shard_shape_[core_id][0] *
                    buffer.shard_spec().shape_in_pages()[1];
        if (num_pages == 0) {
            return;
        }
        dispatch_params.dst_page_index = dispatch_params.buffer_page_mapping->host_page_to_dev_page_mapping_
                                             [dispatch_params.buffer_page_mapping->core_host_page_indices_[core_id][0]];
    } else {
        num_pages = std::min(dispatch_params.total_pages_to_write, dispatch_params.max_pages_per_shard);
        dispatch_params.total_pages_to_write -= num_pages;
    }
    uint32_t curr_page_idx_in_shard = 0;
    uint32_t bank_base_address = buffer.address();
    if (buffer.is_dram()) {
        bank_base_address +=
            buffer.device()->bank_offset(BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }

    while (num_pages != 0) {
        // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
        uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
        dispatch_params.issue_wait =
            dispatch_params.dst_page_index == 0;  // only stall for the first write of the buffer
        if (dispatch_params.issue_wait) {
            // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            data_offset_bytes *= 2;
        }
        uint32_t space_available_bytes = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(dispatch_params.page_size_to_write);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        dispatch_params.pages_per_txn = std::min(num_pages, (uint32_t)num_pages_available);
        dispatch_params.address = bank_base_address + curr_page_idx_in_shard * dispatch_params.page_size_to_write;
        dispatch_params.core = core;

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", dispatch_params.cq_id);

        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        curr_page_idx_in_shard += dispatch_params.pages_per_txn;
        num_pages -= dispatch_params.pages_per_txn;
        dispatch_params.dst_page_index += dispatch_params.pages_per_txn;
    }
}

// Main API to write buffer data
void write_to_device_buffer(
    const void* src,
    Buffer& buffer,
    const BufferRegion& region,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    CoreType dispatch_core_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();
    const BufferDispatchConstants buf_dispatch_constants =
        generate_buffer_dispatch_constants(sysmem_manager, dispatch_core_type, cq_id);

    if (is_sharded(buffer.buffer_layout())) {
        ShardedBufferWriteDispatchParams dispatch_params = initialize_sharded_buf_dispatch_params(
            buffer, cq_id, expected_num_workers_completed, buf_dispatch_constants);
        const auto cores =
            get_cores_for_sharded_buffer(dispatch_params.width_split, dispatch_params.buffer_page_mapping, buffer);
        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            write_sharded_buffer_to_core(
                src,
                core_id,
                buffer,
                dispatch_params,
                buf_dispatch_constants,
                sub_device_ids,
                cores[core_id],
                dispatch_core_type);
        }
    } else {
        InterleavedBufferWriteDispatchParams dispatch_params = initialize_interleaved_buf_dispatch_params(
            buffer, buf_dispatch_constants, cq_id, expected_num_workers_completed, region);
        write_interleaved_buffer_to_device(
            src, dispatch_params, buffer, buf_dispatch_constants, sub_device_ids, dispatch_core_type);
    }
}

// ====== Utility Functions for Reads ======

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferReadDispatchParams initialize_sharded_buf_read_dispatch_params(
    Buffer& buffer, uint32_t cq_id, tt::stl::Span<const uint32_t> expected_num_workers_completed) {
    // Note that the src_page_index is the device page idx, not the host page idx
    // Since we read core by core we are reading the device pages sequentially
    ShardedBufferReadDispatchParams dispatch_params;
    dispatch_params.cq_id = cq_id;
    dispatch_params.device = buffer.device();
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    dispatch_params.src_page_index = 0;
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.num_total_pages = buffer.num_pages();
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

BufferReadDispatchParams initialize_interleaved_buf_read_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    if (buffer.is_valid_partial_region(region)) {
        TT_FATAL(
            region.offset % buffer.page_size() == 0,
            "Offset {} must be a multiple of the buffer page size {}.",
            region.offset,
            buffer.page_size());
        TT_FATAL(
            region.size % buffer.page_size() == 0,
            "Size {} must be a multiple of the buffer page size {}.",
            region.size,
            buffer.page_size());
        TT_FATAL(
            (region.size + region.offset) <= buffer.size(),
            "(Size + offset) {} must be <= the buffer size {}.",
            region.size + region.offset,
            buffer.size());
    }

    BufferReadDispatchParams dispatch_params;
    dispatch_params.pages_per_txn = region.size / buffer.page_size();
    dispatch_params.src_page_index = region.offset / buffer.page_size();
    dispatch_params.cq_id = cq_id;
    dispatch_params.device = buffer.device();
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

// Issue dispatch commands for forwarding device buffer data to the Completion Queue
template <typename T>
void issue_read_buffer_dispatch_command_sequence(
    Buffer& buffer, T& dispatch_params, tt::stl::Span<const SubDeviceId> sub_device_ids, CoreType dispatch_core_type) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_worker_counters = sub_device_ids.size();
    // accounts for padding
    uint32_t cmd_sequence_sizeB =
        hal.get_alignment(HalMemType::HOST) *
            num_worker_counters +              // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        hal.get_alignment(HalMemType::HOST) +  // CQ_PREFETCH_CMD_STALL
        hal.get_alignment(
            HalMemType::HOST) +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    uint32_t dispatch_message_base_addr =
        dispatch_constants::get(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier + prefetch stall for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = sub_device_ids[i].to_index();
        uint32_t dispatch_message_addr =
            dispatch_message_base_addr +
            dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
        command_sequence.add_dispatch_wait(
            false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
    }
    auto offset_index = sub_device_ids[last_index].to_index();
    uint32_t dispatch_message_addr =
        dispatch_message_base_addr +
        dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        true, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);

    bool flush_prefetch = false;
    command_sequence.add_dispatch_write_host(
        flush_prefetch, dispatch_params.pages_per_txn * dispatch_params.padded_page_size, false);

    // Buffer layout specific logic
    if constexpr (std::is_same_v<T, ShardedBufferReadDispatchParams>) {
        const CoreCoord virtual_core =
            buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
        command_sequence.add_prefetch_relay_linear(
            dispatch_params.device->get_noc_unicast_encoding(dispatch_downstream_noc, virtual_core),
            dispatch_params.padded_page_size * dispatch_params.pages_per_txn,
            dispatch_params.address);
    } else {
        command_sequence.add_prefetch_relay_paged(
            buffer.is_dram(),
            dispatch_params.src_page_index,
            dispatch_params.address,
            dispatch_params.padded_page_size,
            dispatch_params.pages_per_txn);
    }

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level functions to copy device buffers into the completion queue
void copy_sharded_buffer_from_core_to_completion_queue(
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferReadDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    uint32_t pages_per_txn;

    if (dispatch_params.width_split) {
        pages_per_txn = dispatch_params.buffer_page_mapping->core_shard_shape_[core_id][0] *
                        buffer.shard_spec().shape_in_pages()[1];
    } else {
        pages_per_txn = std::min(dispatch_params.num_total_pages, dispatch_params.max_pages_per_shard);
        dispatch_params.num_total_pages -= pages_per_txn;
    }
    uint32_t bank_base_address = buffer.address();
    if (buffer.is_dram()) {
        bank_base_address +=
            buffer.device()->bank_offset(BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }

    dispatch_params.pages_per_txn = pages_per_txn;

    if (dispatch_params.pages_per_txn > 0) {
        if (dispatch_params.width_split) {
            uint32_t host_page = dispatch_params.buffer_page_mapping->core_host_page_indices_[core_id][0];
            dispatch_params.src_page_index =
                dispatch_params.buffer_page_mapping->host_page_to_dev_page_mapping_[host_page];
            dispatch_params.unpadded_dst_offset = host_page * buffer.page_size();
        } else {
            dispatch_params.unpadded_dst_offset = dispatch_params.src_page_index * buffer.page_size();
        }
        dispatch_params.address = bank_base_address;
        dispatch_params.core = core;
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    }
}

void copy_interleaved_buffer_to_completion_queue(
    BufferReadDispatchParams& dispatch_params,
    Buffer& buffer,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    if (dispatch_params.pages_per_txn > 0) {
        uint32_t bank_base_address = buffer.address();

        // Only 8 bits are assigned for the page offset in CQPrefetchRelayPagedCmd
        // To handle larger page offsets move bank base address up and update page offset to be relative to the new
        // bank address
        if (dispatch_params.src_page_index > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
            const uint32_t num_banks = dispatch_params.device->num_banks(buffer.buffer_type());
            const uint32_t num_pages_per_bank = dispatch_params.src_page_index / num_banks;
            bank_base_address += num_pages_per_bank * buffer.aligned_page_size();
            dispatch_params.src_page_index = dispatch_params.src_page_index % num_banks;
        }
        dispatch_params.address = bank_base_address;
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    }
}

// Functions used to copy buffer data from completion queue into user space
std::shared_ptr<tt::tt_metal::detail::CompletionReaderVariant> generate_sharded_buffer_read_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    // Increment the src_page_index after the Read Buffer Descriptor has been populated
    // for the current core/txn
    auto initial_src_page_index = dispatch_params.src_page_index;
    dispatch_params.src_page_index += dispatch_params.pages_per_txn;
    return std::make_shared<tt::tt_metal::detail::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::detail::ReadBufferDescriptor>,
        buffer.buffer_layout(),
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.pages_per_txn,
        initial_src_page_index,
        dispatch_params.buffer_page_mapping);
}

std::shared_ptr<tt::tt_metal::detail::CompletionReaderVariant> generate_interleaved_buffer_read_descriptor(
    void* dst, BufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    return std::make_shared<tt::tt_metal::detail::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::detail::ReadBufferDescriptor>,
        buffer.buffer_layout(),
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.pages_per_txn,
        dispatch_params.src_page_index);
}

void copy_completion_queue_data_into_user_space(
    const detail::ReadBufferDescriptor& read_buffer_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint32_t cq_id,
    SystemMemoryManager& sysmem_manager,
    volatile bool& exit_condition) {
    const auto& [buffer_layout, page_size, padded_page_size, buffer_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id] =
        read_buffer_descriptor;
    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    while (remaining_bytes_to_read != 0) {
        uint32_t completion_queue_write_ptr_and_toggle =
            sysmem_manager.completion_queue_wait_front(cq_id, exit_condition);

        if (exit_condition) {
            break;
        }

        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
        uint32_t completion_q_read_toggle = sysmem_manager.get_completion_queue_read_toggle(cq_id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue = sysmem_manager.get_completion_queue_limit(cq_id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered = div_up(bytes_xfered, dispatch_constants::TRANSFER_PAGE_SIZE);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_page_mapping == nullptr) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if (page_size == padded_page_size) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::Cluster::instance().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::Cluster::instance().read_sysmem(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        dev_page_id++;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = buffer_page_mapping->dev_page_to_host_page_mapping_[dev_page_id];
                    dev_page_id++;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::Cluster::instance().read_sysmem(
                    (char*)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        sysmem_manager.completion_queue_pop_front(num_pages_xfered, cq_id);
    }
}

template void issue_buffer_dispatch_command_sequence<InterleavedBufferWriteDispatchParams>(
    const void*, Buffer&, InterleavedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_buffer_dispatch_command_sequence<ShardedBufferWriteDispatchParams>(
    const void*, Buffer&, ShardedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

template void issue_read_buffer_dispatch_command_sequence<BufferReadDispatchParams>(
    Buffer&, BufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_read_buffer_dispatch_command_sequence<ShardedBufferReadDispatchParams>(
    Buffer&, ShardedBufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
