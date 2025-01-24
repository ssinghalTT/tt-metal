// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks sampling_multicore_interleaved(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const std::vector<uint16_t>& k,
    const std::vector<float>& p,
    const uint32_t seed,
    Tensor& output_tensor);

}  // namespace ttnn::operations::reduction::detail
