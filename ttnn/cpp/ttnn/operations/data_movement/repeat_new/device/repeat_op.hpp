// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
namespace ttnn {

struct RM_REPEAT_STRUCT {
    const uint32_t m_num_repeats;
    const bool m_is_last_dim;
    MemoryConfig m_output_mem_config;

    // Required functions to all tensor op functions
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<SimpleShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};
}  // namespace ttnn
