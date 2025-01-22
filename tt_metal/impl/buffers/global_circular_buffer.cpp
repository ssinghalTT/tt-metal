// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <global_circular_buffer_impl.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include <assert.hpp>
#include <core_coord.hpp>
#include <tt_metal.hpp>
#include <host_api.hpp>
#include <buffer.hpp>
#include <buffer_constants.hpp>
#include <device.hpp>
#include <hal.hpp>
#include <tt_align.hpp>

namespace tt::tt_metal {

namespace v1 {

namespace experimental {

GlobalCircularBuffer::GlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    device_(device), sender_receiver_core_mapping_(sender_receiver_core_mapping), size_(size) {
    TT_FATAL(this->device_ != nullptr, "Device cannot be null");
    uint32_t num_sender_cores = sender_receiver_core_mapping.size();
    uint32_t num_receiver_cores = 0;
    uint32_t max_num_receivers_per_sender = 0;
    std::vector<CoreRange> sender_cores;
    sender_cores.reserve(num_sender_cores);
    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
        num_receiver_cores += receiver_cores.num_cores();
        sender_cores.emplace_back(sender_core);
        this->receiver_cores_ = this->receiver_cores_.merge(receiver_cores);
        max_num_receivers_per_sender = std::max(max_num_receivers_per_sender, receiver_cores.num_cores());
    }
    this->sender_cores_ = CoreRangeSet(sender_cores);
    TT_FATAL(num_sender_cores == this->sender_cores_.num_cores(), "Duplicate sender cores found");
    TT_FATAL(num_receiver_cores == this->receiver_cores_.num_cores(), "Duplicate receiver cores found");
    this->all_cores_ = this->sender_cores_.merge(this->receiver_cores_);
    TT_FATAL(this->all_cores_.num_cores() == num_sender_cores + num_receiver_cores, "Duplicate cores found");
    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

void GlobalCircularBuffer::setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global circular buffer can only be created for L1 buffer types");
    uint32_t num_cores = this->all_cores_.num_cores();

    auto shard_parameters =
        ShardSpecBuffer(this->all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    uint32_t cb_buffer_size = this->size_ * num_cores;
    this->cb_buffer_ = Buffer::create(
        this->device_,
        cb_buffer_size,
        this->size_,
        buffer_type,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_parameters,
        std::nullopt);

    auto l1_alignment = hal.get_alignment(HalMemType::L1);
    // is_sender, receiver_val, fifo_start_addr, fifo_size, fifo_ptr, noc_xy coords, and pages_sent
    constexpr uint32_t num_config_elements = 7;
    uint32_t num_noc_xy_words = 2 * max_num_receivers_per_sender;
    auto cb_config_page_size = tt::align((num_config_elements + num_noc_xy_words) * sizeof(uint32_t), l1_alignment) +
                               2 * max_num_receivers_per_sender * l1_alignment;
    uint32_t cb_config_size = cb_config_page_size * num_cores;
    this->cb_config_buffer_ = Buffer::create(
        this->device_,
        cb_config_size,
        cb_config_page_size,
        buffer_type,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_parameters,
        std::nullopt);

    // Write the config buffer to the device
    // Only block for the slow dispatch case
    auto* device = this->device_;
    device->push_work([device,
                       cb_config_size,
                       cb_config_page_size,
                       num_noc_xy_words,
                       l1_alignment,
                       buffer_address = this->cb_buffer_->address(),
                       cb_config_buffer = this->cb_config_buffer_,
                       size = this->size_,
                       sender_receiver_core_mapping = this->sender_receiver_core_mapping_] {
        auto config_buffer_address = cb_config_buffer->address();
        const auto& core_to_core_id = cb_config_buffer->get_buffer_page_mapping()->core_to_core_id_;
        std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
        uint32_t noc_xy_address = config_buffer_address + num_config_elements * sizeof(uint32_t);
        uint32_t pages_sent_address = tt::align(noc_xy_address + num_noc_xy_words * sizeof(uint32_t), l1_alignment);

        for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
            const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
            uint32_t sender_idx = core_to_core_id.at(sender_core) * cb_config_page_size / sizeof(uint32_t);
            uint32_t num_receivers = receiver_cores.num_cores();
            uint32_t pages_acked_address = pages_sent_address + num_receivers * l1_alignment;
            cb_config_host_buffer[sender_idx++] = 1;
            cb_config_host_buffer[sender_idx++] = receiver_cores.num_cores();
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = size;
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = noc_xy_address;
            cb_config_host_buffer[sender_idx++] = pages_sent_address;

            auto sender_physical_coord = device->worker_core_from_logical_core(sender_core);
            for (uint32_t i = 0; i < receiver_cores_vec.size(); i++) {
                auto receiver_physical_coord = device->worker_core_from_logical_core(receiver_cores_vec[i]);
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;

                uint32_t receiver_idx =
                    core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
                cb_config_host_buffer[receiver_idx++] = 0;
                cb_config_host_buffer[receiver_idx++] = num_receivers;
                cb_config_host_buffer[receiver_idx++] = buffer_address;
                cb_config_host_buffer[receiver_idx++] = size;
                cb_config_host_buffer[receiver_idx++] = buffer_address;
                cb_config_host_buffer[receiver_idx++] = noc_xy_address;
                cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment;
                cb_config_host_buffer[receiver_idx++] = sender_physical_coord.x;
                cb_config_host_buffer[receiver_idx++] = sender_physical_coord.y;
            }
        }
        if (device->using_slow_dispatch()) {
            detail::WriteToBuffer(*cb_config_buffer, cb_config_host_buffer);
            tt::Cluster::instance().l1_barrier(device->id());
        } else {
            EnqueueWriteBuffer(device->command_queue(), cb_config_buffer, cb_config_host_buffer.data(), false);
        }
    });
}

const Buffer& GlobalCircularBuffer::cb_buffer() const { return *this->cb_buffer_; }

const CoreRangeSet& GlobalCircularBuffer::sender_cores() const { return this->sender_cores_; }

const CoreRangeSet& GlobalCircularBuffer::receiver_cores() const { return this->receiver_cores_; }

const CoreRangeSet& GlobalCircularBuffer::all_cores() const { return this->all_cores_; }

DeviceAddr GlobalCircularBuffer::buffer_address() const { return this->cb_buffer_->address(); }

DeviceAddr GlobalCircularBuffer::config_address() const { return this->cb_config_buffer_->address(); }

uint32_t GlobalCircularBuffer::size() const { return this->size_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBuffer::sender_receiver_core_mapping() const {
    return this->sender_receiver_core_mapping_;
}

}  // namespace experimental

}  // namespace v1

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::v1::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::v1::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
