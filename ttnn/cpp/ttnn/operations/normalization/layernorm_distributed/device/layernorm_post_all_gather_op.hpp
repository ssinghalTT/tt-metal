// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "layernorm_distributed_types.hpp"

using namespace tt::constants;

namespace ttnn::operations::normalization {

operation::ProgramWithCallbacks layernorm_post_allgather_multi_core(
    const Tensor &a,
    const Tensor &stats,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    LayerNormDistributedType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config);

struct LayerNormPostAllGather {
    LayerNormDistributedType norm_type;
    float eps;
    MemoryConfig memory_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

}  // namespace ttnn::operations::normalization
