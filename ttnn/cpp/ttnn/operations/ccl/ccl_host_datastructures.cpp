// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace ccl {

std::size_t EriscDatamoverConfig::get_eth_channel_sync_size_bytes() { return eth_channel_sync_size_bytes; }

uint32_t EriscDatamoverConfig::get_edm_handshake_address() { return usable_l1_base_address; }

std::size_t EriscDatamoverConfig::get_semaphores_region_size(std::size_t num_edm_channels) {
    return (num_edm_channels * semaphore_size);
}
std::size_t EriscDatamoverConfig::get_semaphores_region_start_offset(std::size_t num_edm_channels) {
    return handshake_location_size + edm_receiver_first_level_ack_source_word_size;
}
uint32_t EriscDatamoverConfig::get_semaphores_base_address(std::size_t num_edm_channels) {
    return usable_l1_base_address + get_semaphores_region_start_offset(num_edm_channels);
}
uint32_t EriscDatamoverConfig::get_buffers_region_start_offset(std::size_t num_edm_channels) {
    return get_semaphores_region_start_offset(num_edm_channels) + get_semaphores_region_size(num_edm_channels);
}
std::size_t EriscDatamoverConfig::get_eth_word_size() { return eth_word_size_bytes; }
uint32_t EriscDatamoverConfig::get_buffers_base_address(std::size_t num_edm_channels) {
    uint32_t base_address = tt::round_up(usable_l1_base_address + get_buffers_region_start_offset(num_edm_channels), eth_word_size_bytes);
    TT_ASSERT(base_address % eth_word_size_bytes == 0);
    return base_address;
}
std::size_t EriscDatamoverConfig::compute_overheads_per_channel_buffer(ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode) {
    std::size_t per_buffer_overhead = 0;
    per_buffer_overhead += (packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::VARIABLE_SIZE) * packet_header_size_bytes;
    per_buffer_overhead += (enable_merged_payload_and_channel_sync * eth_channel_sync_size_bytes);
    return per_buffer_overhead;
}

uint32_t EriscDatamoverConfig::compute_buffer_size(std::size_t num_edm_channels, std::size_t num_buffers_per_channel, uint32_t page_size, ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode) {
    page_size = std::max<uint32_t>(page_size, eth_word_size_bytes);

    std::size_t per_buffer_overhead = compute_overheads_per_channel_buffer(packet_sizing_mode);

    TT_ASSERT(num_edm_channels > 0);
    std::size_t total_usable_space = total_l1_buffer_space - get_buffers_region_start_offset(num_edm_channels);
    std::size_t l1_per_buffer_region = (total_usable_space / (num_edm_channels * num_buffers_per_channel)) - per_buffer_overhead;
    uint32_t buffer_size = tt::round_down(l1_per_buffer_region, page_size);
    log_trace(tt::LogOp, "total_l1_buffer_space: {}", total_l1_buffer_space);
    log_trace(
        tt::LogOp, "get_buffers_base_address(num_edm_channels): {}", get_buffers_base_address(num_edm_channels));
    log_trace(
        tt::LogOp, "usable buffer space: {}", total_l1_buffer_space - get_buffers_base_address(num_edm_channels));
    log_trace(tt::LogOp, "num_edm_channels: {}", num_edm_channels);
    log_trace(tt::LogOp, "page_size: {}", page_size);

    log_trace(tt::LogOp, "Buffer size: {}", buffer_size);

    TT_ASSERT(buffer_size > 0 && buffer_size % page_size == 0);
    return buffer_size;
}

CCLOpConfig::CCLOpConfig(
    std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors, Topology topology) :
    input_tensors(&input_tensors),
    output_tensors(&output_tensors),
    input_sharded(input_tensors.at(0).is_sharded()),
    output_sharded(output_tensors.at(0).is_sharded()),
    page_size(input_tensors.at(0).buffer()->page_size()),
    shard_grid_size(output_tensors.at(0).is_sharded() ? input_tensors.at(0).shard_spec()->num_cores() : 0),
    topology(topology),
    is_row_major(input_tensors.at(0).get_layout() == Layout::ROW_MAJOR) {
}


uint32_t CCLOpConfig::get_page_size() const { return this->page_size; }

Topology CCLOpConfig::get_topology() const { return this->topology; }

bool CCLOpConfig::is_input_sharded() const { return this->input_sharded; }

bool CCLOpConfig::is_output_sharded() const { return this->output_sharded; }

bool CCLOpConfig::get_shard_grid_size() const { return this->shard_grid_size; }

Tensor const& CCLOpConfig::get_input_tensor(std::size_t i) const { return input_tensors->at(i); }

Tensor const& CCLOpConfig::get_output_tensor(std::size_t i) const { return output_tensors->at(i); }

std::map<string, string> CCLOpConfig::emit_worker_defines() const {

    std::map<string, string> worker_defines;
    if (this->is_row_major) {
        worker_defines["ROW_MAJOR_LAYOUT"] = "1";
    } else {
        worker_defines["TILED_LAYOUT"] = "1";
    }
    if (this->input_sharded) {
        TT_ASSERT(this->output_sharded, "CCL Util functions currently don't  support a mix of input sharded with output interleaved or vice versa");
        worker_defines["SHARDED_MEM_LAYOUT"] = "1";
    } else {
        worker_defines["INTERLEAVED_MEM_LAYOUT"] = "1";
    }

    return worker_defines;
}

} // namespace ccl
} // namespace ttnn
