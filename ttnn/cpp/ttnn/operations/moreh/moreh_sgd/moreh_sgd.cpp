// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sgd.hpp"
namespace ttnn::operations::moreh::moreh_sgd {
std::vector<std::optional<Tensor>> MorehSgd::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    const std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    // const CoreRange core_range,
    const const std::optional<MemoryConfig>& param_out_mem_config,
    const const std::optional<MemoryConfig>& momentum_buffer_out_mem_config,
    const DeviceComputeKernelConfig compute_kernel_config) {
    return ttnn::prim::moreh_sgd(
        param_in,
        grad,
        momentum_buffer_in,
        param_out,
        momentum_buffer_out,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized,
        core_range,
        param_out_mem_config,
        momentum_buffer_out_mem_config,
        compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_sgd
