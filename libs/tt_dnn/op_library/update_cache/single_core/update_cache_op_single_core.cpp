#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks update_cache_single_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t update_idx) {
    Program program{};

    CoreRangeSet core({CoreRange{.start={0, 0}, .end={0, 0}}});

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat interm_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    uint32_t Wt = cache_tensor.shape()[-1] / TILE_WIDTH;

    // Width size after untilize
    uint32_t Wbytes = cache_tensor.shape()[-1] * sizeof(bfloat16);


    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    uint32_t cache_Ht = cache_tensor.shape()[-2] / TILE_HEIGHT, cache_Wt = cache_tensor.shape()[-1] / TILE_WIDTH;
    uint32_t cache_HtWt = cache_Ht * cache_Wt;
    uint32_t B = cache_tensor.shape()[0];
    uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
    uint32_t cache_tile_idx = update_idx / TILE_HEIGHT * Wt;
    tt_metal::Device *device = input_tensor.device();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2 * Wt;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );
    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t interm0_cb_index = 24;
    uint32_t interm1_cb_index = 25;
    uint32_t num_interm_tiles = Wt;
    auto cb_interm0 = tt_metal::CreateCircularBuffers(
        program,
        {interm0_cb_index, interm1_cb_index},
        core,
        num_interm_tiles,
        num_interm_tiles * interm_single_tile_size,
        interm_cb_data_format
    );

    uint32_t interm2_cb_index = 26;
    auto cb_interm2 = tt_metal::CreateCircularBuffers(
        program,
        interm2_cb_index,
        core,
        num_interm_tiles,
        num_interm_tiles * interm_single_tile_size,
        interm_cb_data_format
    );

    // Output is same tensor as input, so cb/tile size is same
    uint32_t output_cb_index = 16;
    uint32_t num_output_tiles = 2 * Wt;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * input_single_tile_size,
        input_cb_data_format
    );


    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)dst_is_dram,
        (uint32_t)src_is_dram,
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index
    };


    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) output_cb_index,
        (std::uint32_t) interm0_cb_index,
        (std::uint32_t) interm1_cb_index,
        (std::uint32_t) interm2_cb_index
    };

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        interm0_cb_index,
        interm1_cb_index,
        interm2_cb_index,
        output_cb_index,
        B,
        Wt
    };

    auto eltwise_unary_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/update_cache.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            dst_buffer->address(),
            src_buffer->address(),
            Wt, B, cache_HtWt, cache_tile_idx, 0
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            Wt, B, cache_HtWt, Wbytes, cache_tile_idx, tile_update_offset
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(1);

        auto dst_buffer = input_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = src_buffer->address();
            SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


operation::ProgramWithCallbacks fill_cache_single_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t batch_idx, const uint32_t update_idx) {
    Program program{};

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);


    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    uint32_t cache_Ht = cache_tensor.shape()[-2] / TILE_HEIGHT, cache_Wt = cache_tensor.shape()[-1] / TILE_WIDTH;
    uint32_t cache_HtWt = cache_Ht * cache_Wt;
    uint32_t update_idxt = update_idx / TILE_HEIGHT;
    uint32_t start_idx = batch_idx * cache_HtWt + update_idxt * cache_Wt;
    tt_metal::Device *device = input_tensor.device();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = src0_cb_index;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            num_tiles, 0
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_tiles, start_idx
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(1);

        auto dst_buffer = input_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
