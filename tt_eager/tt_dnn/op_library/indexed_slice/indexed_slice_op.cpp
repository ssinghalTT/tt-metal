// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/indexed_slice/indexed_slice_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void IndexedSlice::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(1);
    TT_FATAL(this->dim == 0, "Currently only supporting batch dimension");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Index Slice need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to Index Slice need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Index Slice does not currently support sharding");
}

std::vector<Shape> IndexedSlice::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& batch_ids = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(0);
    std::vector<uint32_t>new_shape_vec(input_tensor.get_legacy_shape().rank());
    new_shape_vec[0] = batch_ids.get_legacy_shape()[0];
    for(int dim=1; dim<input_tensor.get_legacy_shape().rank(); dim++) {
        new_shape_vec[dim] = input_tensor.get_legacy_shape()[dim];
    }
    return {Shape(new_shape_vec)};
}

std::vector<Tensor> IndexedSlice::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(1);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks IndexedSlice::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& batch_ids = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    return indexed_slice_multi_core(batch_ids, input_tensor, output_tensor);
}


tt::stl::reflection::Attributes IndexedSlice::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor indexed_slice(const Tensor &batch_ids, const Tensor& input, const MemoryConfig& output_mem_config, std::int64_t dim) {
    return operation::run_without_autoformat(IndexedSlice{output_mem_config, dim}, {batch_ids, input}).at(0);
}


}  // namespace tt_metal

}  // namespace tt
