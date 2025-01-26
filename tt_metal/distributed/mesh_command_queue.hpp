// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mesh_device.hpp>
#include <tt-metalium/command_queue_interface.hpp>

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/distributed/mesh_workload.hpp"

namespace tt::tt_metal::distributed {

class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
private:
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id);
    void populate_virtual_program_dispatch_core();
    void populate_dispatch_core_type();
    CoreCoord virtual_program_dispatch_core() const;
    CoreType dispatch_core_type() const;
    // Helper functions for reading and writing individual shards
    void write_shard_to_device(
        std::shared_ptr<Buffer>& shard_view,
        const void* src,
        std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids);
    void read_shard_from_device(
        std::shared_ptr<Buffer>& shard_view,
        void* dst,
        std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids);
    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(
        MeshBuffer& buffer,
        const void* src,
        std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids);
    void read_sharded_buffer(
        MeshBuffer& buffer,
        void* dst,
        std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids);
    tt::tt_metal::WorkerConfigBufferMgr config_buffer_mgr_;
    LaunchMessageRingBufferState worker_launch_message_buffer_state_;
    uint32_t expected_num_workers_completed_ = 0;
    MeshDevice* mesh_device_;
    uint32_t id_;
    CoreCoord dispatch_core_;
    CoreType dispatch_core_type_;

public:
    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id);
    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking);
    // MeshBuffer Write APIs
    void enqueue_write_shard(
        std::shared_ptr<MeshBuffer>& mesh_buffer, void* host_data, const Coordinate& coord, bool blocking);
    void enqueue_write_shard_to_sub_grid(
        MeshBuffer& buffer, void* host_data, const LogicalDeviceRange& device_range, bool blocking);
    void enqueue_write_mesh_buffer(const std::shared_ptr<MeshBuffer>& buffer, void* host_data, bool blocking);
    // MeshBuffer Read APIs
    void enqueue_read_shard(
        void* host_data, const std::shared_ptr<MeshBuffer>& mesh_buffer, const Coordinate& coord, bool blocking);
    void enqueue_read_mesh_buffer(void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking);
    void finish();
};

}  // namespace tt::tt_metal::distributed
