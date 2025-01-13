#pragma once

#include <cstdint>
#include <cstdio>
#include "buffers/circular_buffer_types.hpp"
#include "common/base_types.hpp"
#include "common/constants.hpp"
#include "common/core_coord.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/work_split.hpp"
#include "kernels/kernel_types.hpp"
#include "kernels/runtime_args_data.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn {

namespace operations::multiplyadd {

struct MultiplyAddDeviceOperation {
    struct operation_attributes_t {};
    struct tensor_args_t {
        const ttnn::Tensor& input_tensor1;
        const ttnn::Tensor& input_tensor2;
        const ttnn::Tensor& input_tensor3;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using shape_return_value_t = ttnn::Shape;

    struct MultiCore {
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
            tensor_return_value_t& tensor_return_value) {
            const ttnn::Tensor& inputTensor1 = tensor_args.input_tensor1;
            const ttnn::Tensor& inputTensor2 = tensor_args.input_tensor2;
            const ttnn::Tensor& inputTensor3 = tensor_args.input_tensor3;

            uint32_t input_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

            uint32_t num_tiles = inputTensor1.volume() / tt::constants::TILE_HW;

            Buffer* src0_buffer = inputTensor1.buffer();
            Buffer* src1_buffer = inputTensor2.buffer();
            Buffer* src2_buffer = inputTensor3.buffer();
            Buffer* dst_buffer = tensor_return_value.buffer();

            uint32_t src0_cb_index = tt::CBIndex::c_0;
            uint32_t src1_cb_index = tt::CBIndex::c_1;
            uint32_t src2_cb_index = tt::CBIndex::c_2;
            uint32_t dst0_cb_index = tt::CBIndex::c_3;
            uint32_t dst1_cb_index = tt::CBIndex::c_4;

            uint32_t num_input_tiles = 2;
            std::optional<ShardSpec> shard_spec = std::nullopt;
            if (inputTensor1.memory_config().is_sharded()) {
                shard_spec = inputTensor1.shard_spec().value();
            }
            if (inputTensor2.memory_config().is_sharded()) {
                shard_spec = inputTensor2.shard_spec().value();
            }
            if (inputTensor3.memory_config().is_sharded()) {
                shard_spec = inputTensor3.shard_spec().value();
            }
            if (tensor_return_value.memory_config().is_sharded()) {
                shard_spec = tensor_return_value.shard_spec().value();
            }

            Device* device = inputTensor1.device();
            Program program = CreateProgram();
            CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;

            CoreRangeSet all_cores;
            uint32_t num_cores;
            CoreRangeSet core_group_1;
            CoreRangeSet core_group_2;
            uint32_t num_tiles_per_core_group_1;
            uint32_t num_tiles_per_core_group_2;

            if (shard_spec.has_value()) {
                // Guaranteed that one core is getting full size shard
                uint32_t num_tiles_per_shard =
                    shard_spec.value().shape[0] * shard_spec.value().shape[1] / tt::constants::TILE_HW;
                num_input_tiles = num_tiles_per_shard;
                std::tie(
                    num_cores,
                    all_cores,
                    core_group_1,
                    core_group_2,
                    num_tiles_per_core_group_1,
                    num_tiles_per_core_group_2) =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles / num_tiles_per_shard);

                num_tiles_per_core_group_1 *= num_tiles_per_shard;
            } else {
                std::tie(
                    num_cores,
                    all_cores,
                    core_group_1,
                    core_group_2,
                    num_tiles_per_core_group_1,
                    num_tiles_per_core_group_2) =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles / num_input_tiles);
                num_tiles_per_core_group_1 *= num_input_tiles;
            }

            CircularBufferConfig cb_src0_config =
                CircularBufferConfig(num_input_tiles * input_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, input_tile_size);
            if (inputTensor1.memory_config().is_sharded()) {
                cb_src0_config = cb_src0_config.set_globally_allocated_address(*src0_buffer);
            }
            CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

            CircularBufferConfig cb_src1_config =
                CircularBufferConfig(num_input_tiles * input_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, input_tile_size);
            if (inputTensor2.memory_config().is_sharded()) {
                cb_src1_config = cb_src1_config.set_globally_allocated_address(*src1_buffer);
            }
            CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

            CircularBufferConfig cb_src2_config =
                CircularBufferConfig(num_input_tiles * input_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src2_cb_index, input_tile_size);
            if (inputTensor3.memory_config().is_sharded()) {
                cb_src2_config = cb_src2_config.set_globally_allocated_address(*src2_buffer);
            }
            CBHandle cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);

            CircularBufferConfig cb_dst0_config =
                CircularBufferConfig(num_input_tiles * input_tile_size, {{dst0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(dst0_cb_index, input_tile_size);
            CBHandle cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst0_config);

            CircularBufferConfig cb_dst1_config =
                CircularBufferConfig(num_input_tiles * input_tile_size, {{dst1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(dst1_cb_index, input_tile_size);
            if (tensor_return_value.memory_config().is_sharded()) {
                cb_dst1_config = cb_dst1_config.set_globally_allocated_address(*dst_buffer);
            }
            CBHandle cb_dst1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst1_config);

            KernelHandle reader_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/dataflow/reader.cpp",
                all_cores,
                ReaderDataMovementConfig());

            KernelHandle writer_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/dataflow/writer.cpp",
                all_cores,
                WriterDataMovementConfig());

            KernelHandle compute_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/compute/fpu.cpp",
                all_cores,
                tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};
                uint32_t num_tiles_per_core = 0;
                if (core_group_1.contains(core)) {
                    num_tiles_per_core = num_tiles_per_core_group_1;
                } else if (core_group_2.contains(core)) {
                    num_tiles_per_core = num_tiles_per_core_group_2;
                } else {
                    TT_ASSERT(false, "Core not in specified core ranges");
                }
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_buffer->address(),
                     src1_buffer->address(),
                     src2_buffer->address(),
                     num_tiles_per_core,
                     num_tiles_written,
                     inputTensor1.memory_config().is_sharded(),
                     inputTensor2.memory_config().is_sharded(),
                     inputTensor3.memory_config().is_sharded()});
                tt::tt_metal::SetRuntimeArgs(program, compute_id, core, {num_tiles_per_core});
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    writer_id,
                    core,
                    {dst_buffer->address(),
                     num_tiles_per_core,
                     num_tiles_written,
                     tensor_return_value.memory_config().is_sharded()});

                num_tiles_written += num_tiles_per_core;
            }
            if (inputTensor1.memory_config().is_sharded()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program, cb_src0, *src0_buffer, num_tiles_per_core_group_1 * input_tile_size);
            }
            if (inputTensor2.memory_config().is_sharded()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program, cb_src1, *src1_buffer, num_tiles_per_core_group_1 * input_tile_size);
            }
            if (inputTensor3.memory_config().is_sharded()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program, cb_src2, *src2_buffer, num_tiles_per_core_group_1 * input_tile_size);
            }
            if (tensor_return_value.memory_config().is_sharded()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program, cb_dst1, *dst_buffer, num_tiles_per_core_group_1 * input_tile_size);
            }

            return {
                std::move(program),
                {.reader_kernel_id = reader_id,
                 .compute_kernel_id = compute_id,
                 .writer_kernel_id = writer_id,
                 .num_cores_x = num_cores_x,
                 .num_cores_y = num_cores_y}};
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            Program& program = cached_program.program;
            KernelHandle& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
            KernelHandle& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
            KernelHandle& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
            size_t& num_cores_x = cached_program.shared_variables.num_cores_x;
            size_t& num_cores_y = cached_program.shared_variables.num_cores_y;
            uint32_t num_cores = num_cores_x * num_cores_y;

            const ttnn::Tensor& input_tensor1 = tensor_args.input_tensor1;
            const ttnn::Tensor& input_tensor2 = tensor_args.input_tensor2;
            const ttnn::Tensor& input_tensor3 = tensor_args.input_tensor3;
            ttnn::Tensor& output_tensor = tensor_return_value;

            Buffer* src0_buffer = input_tensor1.buffer();
            Buffer* src1_buffer = input_tensor2.buffer();
            Buffer* src2_buffer = input_tensor3.buffer();

            Buffer* dst_buffer = output_tensor.buffer();

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_x; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};
                {
                    RuntimeArgsData& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src0_buffer->address();
                    runtime_args[1] = src1_buffer->address();
                    runtime_args[2] = src2_buffer->address();
                }

                {
                    RuntimeArgsData& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_buffer->address();
                }
            }
        }
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t&) {
        return MultiCore{};
    }

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        validate_on_program_cache_hit(attributes, tensor_args);
    }

    static void validate_on_program_cache_hit(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {};

    static shape_return_value_t compute_output_shapes(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return tensor_args.input_tensor2.shape();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        shape_return_value_t output_shape = compute_output_shapes(attributes, tensor_args);
        return create_device_tensor(
            output_shape,
            tensor_args.input_tensor2.dtype(),
            tensor_args.input_tensor2.layout(),
            tensor_args.input_tensor2.device(),
            tensor_args.input_tensor2.memory_config());
    }
    static std::tuple<MultiplyAddDeviceOperation::operation_attributes_t, MultiplyAddDeviceOperation::tensor_args_t>
    invoke(const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const ttnn::Tensor& input_tensor3) {
        return {operation_attributes_t{}, tensor_args_t{input_tensor1, input_tensor2, input_tensor3}};
    };
};

}  // namespace operations::multiplyadd
}  // namespace ttnn
namespace ttnn::prim {
constexpr auto multiplyadd =
    ttnn::register_operation<"ttnn::prim::multiplyadd", ttnn::operations::multiplyadd::MultiplyAddDeviceOperation>();
}
