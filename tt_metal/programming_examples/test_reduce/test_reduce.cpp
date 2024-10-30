// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    Device *device = CreateDevice(0);

    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t H = 32;
    constexpr uint32_t C = 512;
    constexpr uint32_t ROW_SIZE = C * 2;
    constexpr uint32_t REDUCTION_MULTI = 8;
    constexpr uint32_t in_ntiles_c = C / tt::constants::TILE_WIDTH;
    constexpr uint32_t in_nblocks_c = in_ntiles_c / REDUCTION_MULTI;
    constexpr uint32_t nsticks_per_core = 1;

    ShardSpecBuffer in_shard_spec = ShardSpecBuffer(
        CoreRangeSet(std::set<CoreRange>({CoreRange(core)})),
        {H, C},
        ShardOrientation::ROW_MAJOR,
        false,
        {1, C},
        {H, 1});
    ShardedBufferConfig in_buf_config{
        .device = device,
        .size = H * ROW_SIZE,
        .page_size = ROW_SIZE,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = in_shard_spec,
    };
    std::shared_ptr<Buffer> in_buf = CreateBuffer(in_buf_config);

    // Raw input data
    auto raw_in_cb_id = tt::CB::c_in2;
    uint32_t raw_in_cb_npages = H;
    uint32_t raw_in_cb_pagesize = ROW_SIZE;
    CircularBufferConfig raw_in_cb_config =
        CircularBufferConfig(raw_in_cb_npages * raw_in_cb_pagesize, {{raw_in_cb_id, tt::DataFormat::Float16_b}})
            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
            .set_globally_allocated_address(*in_buf);
    auto raw_in_cb = tt::tt_metal::CreateCircularBuffer(program, core, raw_in_cb_config);

    // Data from reader to compute kernel
    constexpr uint32_t in_cb_index = CB::c_in0;
    constexpr uint32_t in_cb_pagesize = tt::constants::TILE_HW * REDUCTION_MULTI * 2;
    constexpr uint32_t in_cb_npages = 2;
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{in_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in_cb_index, in_cb_pagesize);
    CBHandle cb_input = tt_metal::CreateCircularBuffer(program, core, cb_in_config);

    // Output buffer
    constexpr uint32_t scalar_cb_index = CB::c_in4;
    CircularBufferConfig cb_scalar_config =
        CircularBufferConfig(tt::constants::TILE_HW * 2, {{scalar_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(scalar_cb_index, tt::constants::TILE_HW * 2);
    CBHandle cb_scalar = tt_metal::CreateCircularBuffer(program, core, cb_scalar_config);

    ShardSpecBuffer out_shard_spec = ShardSpecBuffer(
        CoreRangeSet(std::set<CoreRange>({CoreRange(core)})),
        {nsticks_per_core, C},
        ShardOrientation::ROW_MAJOR,
        false,
        {1, C},
        {nsticks_per_core, 1});
    ShardedBufferConfig out_buf_config{
        .device = device,
        .size = nsticks_per_core * ROW_SIZE,
        .page_size = ROW_SIZE,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = out_shard_spec,
    };
    std::shared_ptr<Buffer> out_buf = CreateBuffer(out_buf_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t out_cb_pagesize = ROW_SIZE / in_nblocks_c;
    constexpr uint32_t out_cb_npages = nsticks_per_core * in_nblocks_c;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(out_cb_pagesize * out_cb_npages, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, out_cb_pagesize)
            .set_globally_allocated_address(*out_buf);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    constexpr uint32_t window_h = 3;
    constexpr uint32_t window_w = 3;
    constexpr uint32_t in_w = 9;
    constexpr uint32_t pad_w = 1;

    const float scalar = 1.0;
    const uint32_t bf16_scalar_u32 = *reinterpret_cast<const uint32_t *>(&scalar);
    std::vector<uint32_t> data_kernel_args = {
        /*reader_nindices=*/nsticks_per_core,
        /*window_h=*/window_h,
        /*window_w=*/window_w,
        /*pad_w=*/pad_w,
        /*in_nbytes_c=*/ROW_SIZE,
        /*in_w=*/in_w,
        /*reader_id=*/0,
        /*bf16_scalar_u32=*/bf16_scalar_u32,
        /*in_nblocks_c=*/in_nblocks_c,
    };
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/test_reduce/kernels/dataflow/reader_wide.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = data_kernel_args,
        });

    constexpr uint32_t in_ntiles_hw = 1;
    std::vector<uint32_t> compute_kernel_args = {
        /*in_ntiles_c=*/in_ntiles_c,
        /*in_ntiles_hwc=*/in_ntiles_hw * in_ntiles_c,
        /*window_size_hw=*/window_h * window_w,
        /*out_ntiles_c=*/in_ntiles_c,
        /*nsticks_per_core=*/nsticks_per_core,
        /*in_c=*/C,
        /*in_nblocks_c=*/in_nblocks_c,
    };
    std::map<string, string> compute_defines = {
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_COL"},
    };
    KernelHandle reduce_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/test_reduce/kernels/compute/reduce.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines,
        });

    std::vector<uint32_t> input_vec = create_constant_vector_of_bfloat16(H * ROW_SIZE, 1.0f);
    EnqueueWriteBuffer(cq, in_buf, input_vec, false);
    SetRuntimeArgs(program, reader_kernel_id, core, {});
    SetRuntimeArgs(program, reduce_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    Finish(cq);

    input_vec = create_constant_vector_of_bfloat16(H * ROW_SIZE, 2.0f);
    EnqueueWriteBuffer(cq, in_buf, input_vec, false);
    SetRuntimeArgs(program, reader_kernel_id, core, {});
    SetRuntimeArgs(program, reduce_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, out_buf, result_vec, true);
    printf("Result (row 0):\n");
    uint32_t offset = 0;
    for (uint32_t i = 0; i < C / tt::constants::TILE_WIDTH; ++i) {
        if (i % REDUCTION_MULTI == 0) {
            printf("c_block %u:\n", i / REDUCTION_MULTI);
        }
        printf("c_tile %u:\t", i);
        for (uint32_t j = 0; j < tt::constants::TILE_WIDTH; ++j) {
            uint32_t val = result_vec[offset / 2];
            val = ((offset & 1) ? (val >> 16) : (val & 0xFFFF)) << 16;
            printf("%.2f ", *reinterpret_cast<float *>(&val));
            offset += 1;
        }
        printf("\n");
    }
    printf("\n");
    CloseDevice(device);
}
