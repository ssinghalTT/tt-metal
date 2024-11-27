// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_rm_op.hpp"
#include "tt_metal/host_api.hpp"

#include <cstdint>

namespace ttnn {

void RM_RESHAPE_STRUCT::validate(const std::vector<Tensor>& input_tensors) const {
    //Validate the input tensor
    const Tensor& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "This function is for RM->RM");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 or input_tensor_a.get_dtype() == DataType::FLOAT32, "Can only work with bfloat16/float32 or uint32 tensors");
    TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout, "Output tensor must have the same memory layout as input tensor");
}

std::vector<SimpleShape> RM_RESHAPE_STRUCT::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {output_shape.logical_shape()};
}

std::vector<Tensor> RM_RESHAPE_STRUCT::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    //Create the output tensor
    const auto& input_tensor_a = input_tensors.at(0);
    auto mem_config = this->output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = output_shape[0];
        mem_config.shard_spec = shard_spec;
    }
    return {create_device_tensor(output_shape, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), input_tensor_a.device(), mem_config, input_tensor_a.tile())};
}

operation::ProgramWithCallbacks RM_RESHAPE_STRUCT::create_program( const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    return operations::data_movement::rm_reshape::rm_reshape_preparer(input_tensors.at(0), output_tensors.at(0));
}
} // namespace ttnn
