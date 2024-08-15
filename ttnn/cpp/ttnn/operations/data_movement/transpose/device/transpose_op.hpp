// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"


namespace ttnn::operations::data_movement {

enum class TransposeOpDim {
    WH, HC, CN, NH, NW, CW, TMP
};

enum class TransposeOpParallelizationStrategy {
    MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN
};

struct Transpose {
    const TransposeOpDim dim;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    TransposeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
};


}
