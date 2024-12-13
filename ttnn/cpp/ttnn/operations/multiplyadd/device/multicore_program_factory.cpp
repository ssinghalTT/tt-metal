#include <cstdint>
#include "buffers/circular_buffer_types.hpp"
#include "common/core_coord.hpp"
#include "common/tt_backend_api_types.hpp"
#include "kernels/kernel_types.hpp"
#include "kernels/runtime_args_data.hpp"
#include "multiplyadd_device_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "common/work_split.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::mac {
MultiplyAddDeviceOperation::MultiCore::cached_program_t MultiplyAddDeviceOperation::MultiCore::create(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const ttnn::Tensor& inputTensor1 = tensor_args.input_tensor1;
    const ttnn::Tensor& inputTensor2 = tensor_args.input_tensor2;
    const ttnn::Tensor& inputTensor3 = tensor_args.input_tensor3;

    const tt::tt_metal::SimpleShape &shape1 = inputTensor1.get_logical_shape(),
                                    &shape2 = inputTensor2.get_logical_shape(),
                                    &shape3 = inputTensor3.get_logical_shape();

    DataFormat inputDataFormat = datatype_to_dataformat_converter(inputTensor1.get_dtype());

    uint32_t inputTileSize = tt::tt_metal::detail::TileSize(inputDataFormat);

    uint32_t numTiles = inputTensor1.volume() / TILE_HW;

    auto src0_buffer = inputTensor1.buffer();
    auto src1_buffer = inputTensor2.buffer();
    auto src2_buffer = inputTensor3.buffer();
    auto dst_buffer = tensor_return_value.buffer();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    uint32_t dst0_cb_index = tt::CBIndex::c_3;
    uint32_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t num_input_tiles = 2;
    Program program = CreateProgram();
    Device* device = inputTensor1.device();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, numTiles);

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * inputTileSize, {{src0_cb_index, inputDataFormat}})
            .set_page_size(src0_cb_index, inputTileSize);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * inputTileSize, {{src1_cb_index, inputDataFormat}})
            .set_page_size(src1_cb_index, inputTileSize);
    CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    CircularBufferConfig cb_src2_config =
        CircularBufferConfig(num_input_tiles * inputTileSize, {{src2_cb_index, inputDataFormat}})
            .set_page_size(src2_cb_index, inputTileSize);
    CBHandle cb_src_intermediate = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);

    CircularBufferConfig cb_dst0_config =
        CircularBufferConfig(num_input_tiles * inputTileSize, {{dst0_cb_index, inputDataFormat}})
            .set_page_size(dst0_cb_index, inputTileSize);
    CBHandle cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst0_config);

    CircularBufferConfig cb_dst1_config =
        CircularBufferConfig(num_input_tiles * inputTileSize, {{dst1_cb_index, inputDataFormat}})
            .set_page_size(dst1_cb_index, inputTileSize);
    CBHandle cb_dst1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst1_config);

    auto reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/dataflow/reader_interleaved.cpp",
        all_cores,
        ReaderDataMovementConfig());

    auto writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/dataflow/writer_interleaved.cpp",
        all_cores,
        WriterDataMovementConfig());

    auto compute_id = tt_metal::CreateKernel(
        program, "ttnn/cpp/ttnn/operations/multiplyadd/device/kernels/compute/fpu.cpp", all_cores, ComputeConfig());

    uint32_t num_tiles_per_core = numTiles / num_cores;

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
        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src0_buffer->address(),
             src1_buffer->address(),
             src2_buffer->address(),
             num_tiles_per_core,
             num_tiles_written});
        tt_metal::SetRuntimeArgs(program, compute_id, core, {num_tiles_per_core, num_tiles_written});
        tt_metal::SetRuntimeArgs(
            program, writer_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_id,
         .compute_kernel_id = compute_id,
         .writer_kernel_id = writer_id,
         .num_cores_x = num_cores_x,
         .num_cores_y = num_cores_y}};
}

void MultiplyAddDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    Program& program = cached_program.program;
    KernelHandle& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    KernelHandle& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    size_t& num_cores_x = cached_program.shared_variables.num_cores_x;
    size_t& num_cores_y = cached_program.shared_variables.num_cores_y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    const Tensor& input_tensor1 = tensor_args.input_tensor1;
    const Tensor& input_tensor2 = tensor_args.input_tensor2;
    const Tensor& input_tensor3 = tensor_args.input_tensor3;
    Tensor& output_tensor = tensor_return_value;

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

}  // namespace ttnn::operations::mac
