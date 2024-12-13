#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::multiplyadd {

struct MultiplyAddDeviceOperation {
    struct operation_attributes_t {};
    struct tensor_args_t {
        const ttnn::Tensor& input_tensor1;
        const ttnn::Tensor& input_tensor2;
        const ttnn::Tensor& input_tensor3;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using shape_return_value_t = ttnn::Shape;

    struct MultiCore {
        // Takes care only of DRAM interleaved tensors
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle compute_kernel_id;
            KernelHandle writer_kernel_id;
            std::size_t num_cores_x;
            std::size_t num_cores_y;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCore>;

    // Mandatory methods

    // Select the program factory based on the tensor args
    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t& attributes, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t& attributes, const tensor_args_t&);

    // Compute the output shapes based on the tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t& attributes, const tensor_args_t&);

    // Create the output tensors based on the tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attributes, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    // This is the only method that is called by the user
    // The user will be able to call the operation using `tensor_return_value_t output =
    // ttnn::prim::example(input_tensor)` after the op is registered Keep in mind that the the overload with `queue_id`
    // argument will be added automatically for primitive operations So, the user can also call this operation using
    // `tensor_return_value_t output = ttnn::prim::example(queue_id, input_tensor)`
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor1, const Tensor& input_tensor2, const Tensor& input_tensor3);

    // helper methods
private:
    static bool is_tensor_dram_interleaved(const ttnn::Tensor& tensor);
};

}  // namespace operations::multiplyadd
}  // namespace ttnn
namespace ttnn::prim {
constexpr auto multiplyadd =
    ttnn::register_operation<"ttnn::prim::multiplyadd", ttnn::operations::multiplyadd::MultiplyAddDeviceOperation>();
}
