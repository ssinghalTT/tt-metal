// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks bcast_multi_core_w(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& output_tensor,
    ttnn::BcastOpMath bcast_op);

}  // namespace ttnn::operations::data_movement
