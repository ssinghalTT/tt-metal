
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <overloaded.hpp>
#include <tt_metal.hpp>

#include "tt_metal/distributed/mesh_buffer.hpp"

namespace tt::tt_metal::distributed {
namespace {

void validate_mesh_buffer_config(const MeshBufferConfig& config, const MeshDevice& mesh_device) {
    if (std::holds_alternative<ReplicatedBufferConfig>(config)) {
        // Nothing to validate.
        return;
    }

    const auto& sharded_config = std::get<ShardedBufferConfig>(config);
    const auto [global_buffer_height, global_buffer_width] = sharded_config.global_buffer_shape;
    const auto [shard_height, shard_width] = sharded_config.physical_shard_shape();

    TT_FATAL(
        (global_buffer_height % shard_height == 0) and (global_buffer_width % shard_width == 0),
        "Global buffer shape must be aligned with the shard shape: requested buffer shape: ({}, {}), shard "
        "shape: ({}, {})",
        global_buffer_height,
        global_buffer_width,
        shard_height,
        shard_width);

    const auto num_shard_rows = global_buffer_height / shard_height;
    const auto num_shard_cols = global_buffer_width / shard_width;
    auto num_shards = num_shard_rows * num_shard_cols;

    // The following check needs to account for shard orientation. The scaling factor for
    // replication depends on which orientation we shard/replicate to when writing to device.
    const auto& [height_replicated, width_replicated] = sharded_config.replicated_dims();
    if (height_replicated and width_replicated) {
        // Pure replication
        num_shards *= mesh_device.num_cols() * mesh_device.num_rows();
    } else if (height_replicated or width_replicated) {
        // Replication along row or column dim.
        num_shards *=
            ((sharded_config.shard_orientation == ShardOrientation::ROW_MAJOR) * (mesh_device.num_rows()) +
             (sharded_config.shard_orientation == ShardOrientation::COL_MAJOR) * (mesh_device.num_cols()));
    }
    TT_FATAL(
        num_shards <= mesh_device.num_devices(),
        "The sharded tensor does not fit on the Mesh. Num shards in buffer {}, Num Devices {}",
        num_shards,
        mesh_device.num_devices());
}

}  // namespace

std::shared_ptr<MeshBuffer> MeshBuffer::create(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalBufferConfig& device_local_config,
    MeshDevice* mesh_device,
    std::optional<DeviceAddr> address) {
    validate_mesh_buffer_config(mesh_buffer_config, *mesh_device);

    const DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.size; },
            [mesh_device](const ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.physical_shard_shape();
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    std::shared_ptr<MeshBuffer> mesh_buffer;
    if (!address.has_value()) {
        // Rely on the MeshDevice allocator to provide the address for the entire mesh buffer.
        // The address provided to the backing buffer is used as the address for the MeshBuffer object.
        std::shared_ptr<Buffer> backing_buffer = Buffer::create(
            mesh_device,
            device_local_size,
            device_local_config.page_size,
            device_local_config.buffer_type,
            device_local_config.buffer_layout,
            device_local_config.shard_parameters,
            device_local_config.bottom_up);

        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, device_local_size, mesh_device, backing_buffer));
    } else {
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device));
    }

    mesh_buffer->allocate();

    return mesh_buffer;
}

void MeshBuffer::allocate() {
    if (backing_buffer_) {
        TT_FATAL(
            !address_, "The address for a MeshBuffer should not explicitly be initialized when it is being allocated");
        address_ = backing_buffer_->address();
    } else {
        TT_FATAL(address_, "A MeshBuffer should be provided a valid address if its not being allocated");
    }
    buffers_ = std::vector<std::vector<std::shared_ptr<Buffer>>>(
        mesh_device_->num_rows(), std::vector<std::shared_ptr<Buffer>>(mesh_device_->num_cols()));

    auto allocate_device_buffer_at_address = [this](const Coordinate& coord) {
        std::shared_ptr<Buffer> buffer = Buffer::create(
            mesh_device_->get_device(coord.row, coord.col),
            address_,
            device_local_size_,
            device_local_config_.page_size,
            device_local_config_.buffer_type,
            device_local_config_.buffer_layout,
            device_local_config_.shard_parameters,
            device_local_config_.bottom_up);
        return buffer;
    };

    for (int row = 0; row < mesh_device_->num_rows(); row++) {
        for (int col = 0; col < mesh_device_->num_cols(); col++) {
            buffers_[row][col] = allocate_device_buffer_at_address(Coordinate{row, col});
        }
    }
}

std::shared_ptr<Buffer> MeshBuffer::get_device_buffer(const Coordinate& device_coord) {
    TT_FATAL(
        device_coord.row < mesh_device_->num_rows() and device_coord.col < mesh_device_->num_cols(),
        "Logical coordinates must be within the bounds of the mesh: {}, {}, mesh shape: {}, {}",
        device_coord.row,
        device_coord.col,
        mesh_device_->num_rows(),
        mesh_device_->num_cols());
    return buffers_[device_coord.row][device_coord.col];
}

DeviceAddr MeshBuffer::size() const {
    return std::visit(
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.size; },
            [&](const ShardedBufferConfig& config) { return config.global_size; }},
        config_);
}

MeshBufferLayout MeshBuffer::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(config_) ? MeshBufferLayout::REPLICATED
                                                                   : MeshBufferLayout::SHARDED;
}

const ShardedBufferConfig& MeshBuffer::global_shard_spec() const {
    TT_FATAL(
        global_layout() == MeshBufferLayout::SHARDED, "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

uint32_t MeshBuffer::datum_size_bytes() const {
    // Limitation for now.
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query datum size for buffers sharded across the Mesh");
    return this->global_shard_spec().compute_datum_size_bytes();
}

Shape2D MeshBuffer::physical_shard_shape() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query physical shard shape for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    return sharded_config.physical_shard_shape();
}

std::pair<bool, bool> MeshBuffer::replicated_dims() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query replicated dims for buffers sharded across the Mesh");
    return this->global_shard_spec().replicated_dims();
}

}  // namespace tt::tt_metal::distributed
