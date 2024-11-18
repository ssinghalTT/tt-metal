// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ttnn/operations/conv/conv2d/conv2d.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv_transpose2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;
struct ConvTranpose2dOperation{
    static Result invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        Device * device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> output_padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const conv2d::Conv2dConfig>& conv_config_ = std::nullopt
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt
        );

    static Result invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice * device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> output_padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const conv2d::Conv2dConfig>& conv_config_ = std::nullopt
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt
        );
};

}  // namespace conv_transpose2d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn{
    constexpr auto conv_transpose2d = ttnn::register_operation<"ttnn::conv_transpose2d", operations::conv::conv_transpose2d::ConvTranpose2dOperation>();
}
