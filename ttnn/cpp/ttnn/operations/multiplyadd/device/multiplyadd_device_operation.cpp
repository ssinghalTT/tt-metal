#include "multiplyadd_device_operation.hpp"
#include <cstddef>
#include "common/assert.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::multiplyadd {

MultiplyAddDeviceOperation::program_factory_t MultiplyAddDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiCore{};
}

void MultiplyAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        !(is_tensor_dram_interleaved(tensor_args.input_tensor1) &&
          is_tensor_dram_interleaved(tensor_args.input_tensor2) &&
          is_tensor_dram_interleaved(tensor_args.input_tensor3)),
        "All tensors need to be DRAM interleaved");

    validate_on_program_cache_hit(attributes, tensor_args);
}

void MultiplyAddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

MultiplyAddDeviceOperation::shape_return_value_t MultiplyAddDeviceOperation::compute_output_shapes(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor1.shape();
}

MultiplyAddDeviceOperation::tensor_return_value_t MultiplyAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto output_shape = compute_output_shapes(attributes, tensor_args);
    const auto& input_tensor1 = tensor_args.input_tensor1;
    return create_device_tensor(output_shape, input_tensor1.dtype(), input_tensor1.layout(), input_tensor1.device());
}

bool MultiplyAddDeviceOperation::is_tensor_dram_interleaved(const ttnn::Tensor& tensor) {
    return tensor.memory_config().is_dram() && tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED;
}

std::tuple<MultiplyAddDeviceOperation::operation_attributes_t, MultiplyAddDeviceOperation::tensor_args_t>
MultiplyAddDeviceOperation::invoke(
    const Tensor& input_tensor1, const Tensor& input_tensor2, const Tensor& input_tensor3) {
    return {operation_attributes_t{}, tensor_args_t{input_tensor1, input_tensor2, input_tensor3}};
}
}  // namespace ttnn::operations::multiplyadd
