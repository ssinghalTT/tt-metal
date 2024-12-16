/// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"

namespace ttnn {
namespace ccl {
namespace all_gather_detail {

AllGatherAsync create_all_gather_async_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<Device*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<std::shared_ptr<GlobalSemaphore>>& semaphore_handles,
    std::unordered_map<chip_id_t, SubDeviceId>& sub_device_id_map,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle) {
    uint32_t num_devices = devices.size();

    std::optional<Device*> forward_device = std::nullopt;
    std::optional<Device*> backward_device = std::nullopt;
    std::shared_ptr<const GlobalSemaphore> semaphore_handle = nullptr;
    bool persistent_fabric = fabric_handle.has_value();
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            if (!persistent_fabric) {
                semaphore_handle = semaphore_handles.at(i);  // Get raw pointer
            }
            if (i != 0) {
                backward_device = devices.at(i - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            }
        }
    }

    return ttnn::AllGatherAsync{
        forward_device,
        backward_device,
        dim,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore_handle,
        sub_device_id_map,
        fabric_handle};
}
}  // namespace all_gather_detail
}  // namespace ccl

void AllGatherAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
}

std::vector<ttnn::SimpleShape> AllGatherAsync::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_padded_shape();  // TODO: Replace with get_logical_shape()
    shape[this->dim] *= this->ring_size;
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGatherAsync::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto output_tensors = std::vector<Tensor>();
    output_tensors.reserve(1);
    auto tile = input_tensor.get_tensor_spec().tile();
    if (this->output_mem_config.is_sharded()) {
        output_tensors.push_back(create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config,
            tile));
    } else {
        output_tensors = operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config, tile);
    }
    log_debug(tt::LogOp, "DEBUG: output_tensors[0] address: {}", output_tensors.at(0).buffer()->address());
    return output_tensors;
}

operation::ProgramWithCallbacks AllGatherAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");
    return all_gather_async_multi_core_with_workers(
        input_tensors[0],
        this->forward_device,
        this->backward_device,
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore_handle,
        this->fabric_handle);
}

const operation::Hash AllGatherAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<AllGatherAsync>(
        this->dim, this->num_links, this->ring_size, this->ring_index, this->output_mem_config, this->topology);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_async(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::unordered_map<chip_id_t, SubDeviceId> sub_device_id_map,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_gather_async op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    bool persistent_fabric_mode = fabric_handle.has_value();
    std::vector<std::shared_ptr<GlobalSemaphore>> semaphore_handles;
    if (!persistent_fabric_mode) {
        for (const auto& device : devices) {
            auto handle = GlobalSemaphore::create(device, core_grid, 0);
            log_trace(
                tt::LogOp, "Created semaphore handle at address {} for device {}", handle->address(), device->id());
            semaphore_handles.push_back(handle);
        }
        // HACK: assert every handle address is the same
        TT_FATAL(
            std::all_of(
                semaphore_handles.begin(),
                semaphore_handles.end(),
                [&](const auto& handle) { return handle->address() == semaphore_handles.front()->address(); }),
            "[Hack] All semaphore handles should have the same address");
    }

    operation::launch_op(
        [dim,
         num_links,
         num_devices,
         memory_config,
         devices,
         ccl_topology,
         semaphore_handles,
         sub_device_id_map,
         fabric_handle](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::all_gather_detail::create_all_gather_async_struct(
                    input_tensor,
                    dim,
                    num_links,
                    memory_config,
                    devices,
                    ccl_topology,
                    semaphore_handles,
                    sub_device_id_map,
                    fabric_handle),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
