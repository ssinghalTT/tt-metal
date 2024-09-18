// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward {
struct MorehMeanBackwardOperation {
    struct operation_attributes_t {
        const std::vector<int64_t> dims;
        const bool keepdim;
        const std::optional<Shape> input_grad_shape;
        const MemoryConfig output_memory_config;
        // const CoreRange core_range; // unused for now
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& output_grad;
        const std::optional<Tensor>& input_grad;
    };

    using shape_return_value_t = Shape;
    using tensor_return_value_t = Tensor;

    struct MorehMeanBackwardFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::uint32_t num_cores_to_be_used;
            std::uint32_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MorehMeanBackwardFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& output_grad,
        const std::vector<int64_t> dims,
        const bool keepdim,
        const std::optional<Shape>& input_grad_shape,
        const std::optional<Tensor>& input_grad,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_mean_backward

namespace ttnn::prim {
constexpr auto moreh_mean_backward =
    ttnn::register_operation<"ttnn::prim::moreh_mean_backward", ttnn::operations::moreh::moreh_mean_backward::MorehMeanBackwardOperation>();
}
