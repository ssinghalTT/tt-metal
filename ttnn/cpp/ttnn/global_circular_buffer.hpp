// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

struct MultiDeviceGlobalCircularBuffer {
    MultiDeviceGlobalCircularBuffer(MeshDevice* mesh_device);
    std::vector<GlobalCircularBuffer> global_circular_buffers;

    static constexpr auto attribute_names = std::forward_as_tuple("global_circular_buffers");
    const auto attribute_values() const { return std::forward_as_tuple(this->global_circular_buffers); }
};

// Single Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Multi Device APIs
MultiDeviceGlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::global_circular_buffer
