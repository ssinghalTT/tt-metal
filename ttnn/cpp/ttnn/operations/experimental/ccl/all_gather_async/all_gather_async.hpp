// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<SubDeviceId> subdevice_id = std::nullopt,
        bool enable_persistent_fabric_mode = false,
        bool create_semaphore_handles = true);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const ttnn::ccl::Topology topology,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<SubDeviceId> subdevice_id = std::nullopt,
        bool enable_persistent_fabric_mode = false,
        bool create_semaphore_handles = true);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
