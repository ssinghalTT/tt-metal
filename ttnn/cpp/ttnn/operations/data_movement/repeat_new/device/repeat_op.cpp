// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/host_api.hpp"

#include "ttnn/operations/data_movement/repeat_new/device/host/repeat_program_factory.hpp"
#include "ttnn/operations/data_movement/repeat_new/device/repeat_op.hpp"

namespace ttnn {

void RM_REPEAT_STRUCT::validate(const std::vector<Tensor>& input_tensors) const {
    // Validate the input tensor
    const Tensor& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "This function is for RM->RM");
    TT_FATAL(
        input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 or
            input_tensor_a.get_dtype() == DataType::FLOAT32,
        "Can only work with bfloat16/float32 or uint32 tensors");
    // is this relevant?
    TT_FATAL(
        this->m_output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout,
        "Output tensor must have the same memory layout as input tensor");
}

std::vector<SimpleShape> RM_REPEAT_STRUCT::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors.at(0).get_logical_shape();
    output_shape[m_is_last_dim ? -1 : 1] *= m_num_repeats;
    return {output_shape};
}

std::vector<Tensor> RM_REPEAT_STRUCT::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Create the output tensor
    const auto& input_tensor_a = input_tensors.at(0);
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    // is this relevant?
    auto mem_config = this->m_output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = output_shape[0];
        mem_config.shard_spec = shard_spec;
    }
    return {create_device_tensor(
        output_shape, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), input_tensor_a.device(), mem_config)};
}

operation::ProgramWithCallbacks RM_REPEAT_STRUCT::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return operations::data_movement::repeat::rm_repeat_program_factory(
        input_tensors.at(0), m_num_repeats, output_tensors.at(0), m_is_last_dim);
}
}  // namespace ttnn
