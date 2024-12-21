// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllGatherAsync {
    std::optional<Device*> forward_device;
    std::optional<Device*> backward_device;
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    std::optional<std::shared_ptr<const GlobalSemaphore>> semaphore_handle;
    bool enable_persistent_fabric_mode;

    AllGatherAsync(
        std::optional<Device*> forward_device,
        std::optional<Device*> backward_device,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        std::optional<std::shared_ptr<const GlobalSemaphore>> semaphore_handle,
        bool enable_persistent_fabric_mode) :
        forward_device(forward_device),
        backward_device(backward_device),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        ring_index(ring_index),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore_handle(semaphore_handle),
        enable_persistent_fabric_mode(enable_persistent_fabric_mode) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore_handle", semaphore_handle.has_value() ? semaphore_handle.value().get() : nullptr);

        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace ccl {
namespace all_gather_async_detail {
AllGatherAsync create_all_gather_async_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<Device*>& devices,
    const ccl::Topology topology,
    const std::optional<std::vector<GlobalSemaphore>>& semaphore_handles,
    bool enable_persistent_fabric_mode);
}  // namespace all_gather_async_detail
}  // namespace ccl

// All Gather Variants
operation::ProgramWithCallbacks all_gather_async_multi_core_with_workers(
    const Tensor& input_tensor,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::optional<std::shared_ptr<const GlobalSemaphore>>& semaphore_handle_opt,
    bool enable_persistent_fabric_mode);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_async(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<SubDeviceId> subdevice_id = std::nullopt,
    bool enable_persistent_fabric_mode = false,
    bool create_semaphore_handles = true);  // TODO make reference

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
