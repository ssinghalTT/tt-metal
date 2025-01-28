// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"
#include "device/running_statistics_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

inline Tensor mean_NHW(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    ttnn::SmallVector<int> dims = {2, 3};
    Tensor mean_hw = ttnn::mean(input_tensor, dims, true);
    return ttnn::mean(mean_hw, 0, true);
}

Tensor BatchNorm::invoke(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    const float momentum,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    Tensor batch_mean = mean_NHW(input, memory_config);
    Tensor mean_sq = mean_NHW(ttnn::square(input, memory_config), memory_config);
    Tensor batch_var = ttnn::subtract(mean_sq, ttnn::square(batch_mean, memory_config), std::nullopt, memory_config);
    if (training) {
        Tensor stats =
            ttnn::prim::running_statistics(batch_mean, batch_var, momentum, running_mean, running_var, memory_config);
    } else {
        TT_FATAL(
            (running_mean.has_value() && running_var.has_value()),
            "running_mean and running_var must be defined in evaluation mode");
        batch_mean = running_mean.value();
        batch_var = running_var.value();
    }
    return ttnn::prim::batch_norm(input, batch_mean, batch_var, eps, weight, bias, output, memory_config);
}
}  // namespace ttnn::operations::normalization
