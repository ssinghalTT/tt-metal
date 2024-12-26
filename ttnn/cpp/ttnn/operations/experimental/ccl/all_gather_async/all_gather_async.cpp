// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode,
    bool create_semaphore_handles) {
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode,
        create_semaphore_handles);
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode,
    bool create_semaphore_handles) {
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        memory_config,
        num_preferred_links,
        subdevice_id,
        enable_persistent_fabric_mode,
        create_semaphore_handles);
}

}  // namespace ttnn::operations::experimental::ccl
