// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

struct SamplingOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_values_tensor,
        const Tensor& input_indices_tensor,
        const std::vector<uint16_t>& k,
        const std::vector<uint16_t>& p,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const Tensor& input_values_tensor,
        const Tensor& input_indices_tensor,
        const std::vector<uint16_t>& k,
        const std::vector<uint16_t>& p,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::reduction

constexpr auto sampling =
    ttnn::register_operation_with_auto_launch_op<"ttnn::sampling", ttnn::operations::reduction::SamplingOperation>();

}  // namespace ttnn
