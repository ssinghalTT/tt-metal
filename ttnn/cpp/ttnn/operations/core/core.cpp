// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"

#include <utility>

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/trace.hpp>
#include "cpp/ttnn/operations/data_movement/move/move.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    if (distributed::is_multi_device_tensor(tensor)) {
        return transform(tensor, [&](const Tensor& device_tensor) { return unsqueeze_to_4D(device_tensor); });
    }

    const auto tensor_shape = tensor.get_shape();
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank(4);
    return ttnn::reshape(tensor, tensor_shape_4D);
}

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank) {
    auto shape = tensor.get_shape();
    if (shape.rank() != 4) {
        TT_THROW("Tensor has to be of rank 4!");
    }
    if (rank < 1 or rank > 4) {
        TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
    }

    for (auto index = 0; index < 4 - rank; ++index) {
        if (shape[index] != 1) {
            TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
        }
    }

    if (rank == 4) {
        return tensor;
    }
    return ttnn::reshape(tensor, shape.to_rank(rank));
}

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, IDevice* device, const std::optional<MemoryConfig>& memory_config, uint8_t cq_id) {
    auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    if (mem_config.is_sharded() and (device->arch() == tt::ARCH::BLACKHOLE)) {
        auto interleaved_tensor = tensor.to(device, ttnn::DRAM_MEMORY_CONFIG, cq_id);
        return ttnn::interleaved_to_sharded(ttnn::DefaultQueueId, interleaved_tensor, mem_config, std::nullopt);
    } else {
        return tensor.to(device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG), cq_id);
    }
}

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    uint8_t cq_id) {
    auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    // Currently no direct sharded write support in BLACKHOLE due to alignment issue
    if (mem_config.is_sharded() and (mesh_device->arch() == tt::ARCH::BLACKHOLE)) {
        auto interleaved_tensor = tensor.to(mesh_device, ttnn::DRAM_MEMORY_CONFIG, cq_id);
        return ttnn::interleaved_to_sharded(ttnn::DefaultQueueId, interleaved_tensor, mem_config, std::nullopt);
    } else {
        return tensor.to(mesh_device, mem_config, cq_id);
    }
}

ttnn::Tensor allocate_tensor_on_device(
    const SimpleShape& shape,
    DataType data_type,
    Layout layout,
    IDevice* device,
    const std::optional<MemoryConfig>& memory_config) {
    return allocate_tensor_on_device(
        TensorSpec(
            shape, TensorLayout(data_type, PageConfig(layout), memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG))),
        device);
}

ttnn::Tensor allocate_tensor_on_device(
    const SimpleShape& shape,
    DataType data_type,
    Layout layout,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config) {
    return allocate_tensor_on_device(
        TensorSpec(
            shape, TensorLayout(data_type, PageConfig(layout), memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG))),
        mesh_device);
}

ttnn::Tensor allocate_tensor_on_device(const ttnn::TensorSpec& spec, IDevice* device) {
    return tt::tt_metal::allocate_tensor_on_devices(spec, {device});
}

ttnn::Tensor allocate_tensor_on_device(const ttnn::TensorSpec& spec, MeshDevice* mesh_device) {
    return tt::tt_metal::allocate_tensor_on_devices(spec, mesh_device->get_devices());
}

void copy_host_to_device_tensor(const ttnn::Tensor& host_tensor, ttnn::Tensor device_tensor, uint8_t cq_id) {
    tt::tt_metal::write_tensor(std::move(host_tensor), std::move(device_tensor), cq_id);
}

ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking, uint8_t cq_id) {
    // Currently no direct sharded read support in BLACKHOLE due to alignment issue
    if (tensor.is_sharded() and (tensor.device()->arch() == tt::ARCH::BLACKHOLE)) {
        auto interleaved_tensor = ttnn::sharded_to_interleaved(cq_id, tensor, ttnn::DRAM_MEMORY_CONFIG, std::nullopt);
        return interleaved_tensor.cpu(blocking, cq_id);
    } else {
        return tensor.cpu(blocking, cq_id);
    }
}

void deallocate(Tensor& tensor, bool force) { tensor.deallocate(force); }

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::move(input_tensor, memory_config);
}

// Trace APIs - Single Device
uint32_t begin_trace_capture(IDevice* device, const uint8_t cq_id) {
    ZoneScoped;
    uint32_t tid = Trace::next_id();
    device->push_work([device, cq_id, tid]() mutable { device->begin_trace(cq_id, tid); });
    return tid;
}

void end_trace_capture(IDevice* device, const uint32_t tid, const uint8_t cq_id) {
    ZoneScoped;
    device->push_work([device, cq_id, tid]() mutable { device->end_trace(cq_id, tid); });
}

void execute_trace(IDevice* device, const uint32_t tid, const uint8_t cq_id, bool blocking) {
    ZoneScoped;
    // If blocking, ensure that worker thread blocks until trace is completed
    device->push_work([device, cq_id, tid, blocking]() mutable { device->replay_trace(cq_id, tid, blocking); });
    // If blocking, wait until worker threads have completed
    if (blocking) {
        device->synchronize();
    }
}

void release_trace(IDevice* device, const uint32_t tid) {
    device->push_work([device, tid]() mutable { device->release_trace(tid); });
}

// Trace APIs - Multi Device
uint32_t begin_trace_capture(MeshDevice* device, const uint8_t cq_id) {
    ZoneScoped;
    auto workers = device->get_devices();
    uint32_t tid = Trace::next_id();
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid]() mutable { worker->begin_trace(cq_id, tid); });
    }
    return tid;
}

void end_trace_capture(MeshDevice* device, const uint32_t tid, const uint8_t cq_id) {
    ZoneScoped;
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid]() mutable { worker->end_trace(cq_id, tid); });
    }
}

void execute_trace(MeshDevice* device, const uint32_t tid, const uint8_t cq_id, bool blocking) {
    ZoneScoped;
    auto workers = device->get_devices();
    // If blocking, ensure that each worker thread blocks until device-local trace is completed
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid, blocking]() mutable { worker->replay_trace(cq_id, tid, blocking); });
    }
    // If blocking, wait until worker threads have completed
    if (blocking) {
        for (auto& worker : workers) {
            worker->synchronize();
        }
    }
}

void release_trace(MeshDevice* device, const uint32_t tid) {
    ZoneScoped;
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work([worker, tid]() mutable { worker->release_trace(tid); });
    }
}

}  // namespace ttnn::operations::core
