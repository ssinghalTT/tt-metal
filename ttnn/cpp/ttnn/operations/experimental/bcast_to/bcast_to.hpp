// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental {
struct BcastTo {
    static Tensor invoke(
        const Tensor& input,
        const std::vector<int32_t>& sizes,

        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto broadcast_to = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::broadcast_to",
    ttnn::operations::experimental::BcastTo>();
}
