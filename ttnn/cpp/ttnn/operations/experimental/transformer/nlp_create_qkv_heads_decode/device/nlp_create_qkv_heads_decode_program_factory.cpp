// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::operations::experimental::transformer {

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    std::optional<const uint32_t> batch_offset,
    std::optional<const uint32_t> slice_size,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size) {
    bool is_input_sharded = input_tensor.is_sharded();
    if (is_input_sharded) {
        return multi_core_nlp_create_qkv_heads_decode_sharded_input(
            input_tensor,
            num_q_heads,
            num_kv_heads,
            head_dim,
            overlap_qk_coregrid,
            batch_offset,
            slice_size,
            output,
            compute_with_storage_grid_size);
    } else {
        return multi_core_nlp_create_qkv_heads_decode_interleaved_input(
            input_tensor, num_q_heads, num_kv_heads, head_dim, output, compute_with_storage_grid_size);
    }
}

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_interleaved_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size) {
    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device* device = input_tensor.device();

    bool is_dram = input_tensor.memory_config().buffer_type == tt::tt_metal::BufferType::DRAM;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shape = input_tensor.get_legacy_shape();
    auto in_num_tiles = in_shape[-2] * in_shape[-1] / TILE_HW;

    uint32_t q_output_cb_index = CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[0].buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CBIndex::c_17;
    tt_metal::CircularBufferConfig cb_k_output_config =
        tt_metal::CircularBufferConfig(k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[1].buffer());
    auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

    auto v_shard_spec = output[2].shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CBIndex::c_18;
    tt_metal::CircularBufferConfig cb_v_output_config =
        tt_metal::CircularBufferConfig(v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[2].buffer());
    auto cb_v_output = tt_metal::CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t q_base_addr = input_tensor.buffer()->address();

    // We parallize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of a
    // tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        k_output_cb_index,
        v_output_cb_index,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        1,  // read the first phase
        is_dram ? 1 : 0};
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
        q_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    reader_compile_time_args[9] = 2;  // read the second phase
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
        q_cores,
        tt_metal::WriterDataMovementConfig(reader_compile_time_args));

    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : (i - 16) * sub_tile_line_bytes + 512 * element_size;

        const auto& core = cores[i];
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(2);
        reader_runtime_args = {
            in_tile_offset_by_batch,
            q_base_addr,
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id,
         writer_kernel_id,
         num_cores,
         cb_q_output,
         cb_k_output,
         cb_v_output,
         cores,
         element_size,
         sub_tile_line_bytes](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();

            uint32_t src_kv_buffer_addr = 0;

            auto dst_buffer_query = output_tensors.at(0).buffer();
            auto dst_buffer_key = output_tensors.at(1).buffer();
            auto dst_buffer_value = output_tensors.at(2).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
            UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
            UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

            uint32_t q_base_addr = input_tensors[0].buffer()->address();
            uint32_t q_start_addr = q_base_addr;

            for (uint32_t i = 0; i < num_cores; ++i) {
                uint32_t in_tile_offset_by_batch =
                    i < 16 ? i * sub_tile_line_bytes : (i - 16) * sub_tile_line_bytes + 512 * element_size;
                const auto& core = cores[i];
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = in_tile_offset_by_batch;
                runtime_args[1] = q_start_addr;

                auto& runtime_args_writer = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args_writer[0] = in_tile_offset_by_batch;
                runtime_args_writer[1] = q_start_addr;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_sharded_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    std::optional<const uint32_t> batch_offset,
    std::optional<const uint32_t> slice_size,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size) {
    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;
    auto in_num_tiles = in_shard_spec.shape[0] * in_shard_spec.shape[1] / TILE_HW;

    uint32_t q_output_cb_index = CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[0].buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CBIndex::c_17;
    tt_metal::CircularBufferConfig cb_k_output_config =
        tt_metal::CircularBufferConfig(k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[1].buffer());
    auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

    auto v_shard_spec = output[0].shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CBIndex::c_18;
    tt_metal::CircularBufferConfig cb_v_output_config =
        tt_metal::CircularBufferConfig(v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[2].buffer());
    auto cb_v_output = tt_metal::CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t q_base_addr = input_tensor.buffer()->address();

        // cores for q
        uint32_t q_num_cores = q_cores.num_cores(); // number of cores of the output
        auto q_core_grid = q_cores.bounding_box();
        uint32_t q_num_cores_x = q_core_grid.end_coord.x + 1, q_num_cores_y = q_core_grid.end_coord.y + 1;
        const auto &q_cores_vector = grid_to_cores(q_num_cores, q_num_cores_x, q_num_cores_y, true);

        // cores for k
        uint32_t k_num_cores = k_cores.num_cores(); // number of cores of the output
        auto k_core_grid = k_cores.bounding_box();
        const auto &k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

        std::vector<uint32_t> noc_x_coords;
        noc_x_coords.reserve(in_num_cores_x);
        for (uint32_t x = 0; x < in_num_cores_x; ++x) {
            noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
        }
        std::vector<uint32_t> noc_y_coords;
        noc_y_coords.reserve(in_num_cores_y);
        for (uint32_t y = 0; y < in_num_cores_y; ++y) {
            noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
        }

        uint32_t process_qv = 1, process_k = 1;
        // In case of overlapping qk coregrid, we create a single set of kernels for q which also process k and v heads from the input and write to the respective output buffers
        // while if q and k are not overlapped, we create two sets of kernels in different coregrids
        // one set of kernels for q which also process v heads but skips k heads from the input and write to the respective output buffers
        // another set of kernels for k which reads k heads from the input and write to the respective output buffers while skipping q and v heads
        if (!overlap_qk_coregrid) {
            process_qv = 1;
            process_k = 0;
        }

        // We parallize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of a tile respectively)
        std::vector<uint32_t> q_reader_compile_time_args = {
            (std::uint32_t) element_size,
            (std::uint32_t) sub_tile_line_bytes,
            q_output_cb_index,
            k_output_cb_index,
            v_output_cb_index,
            head_size,
            num_q_heads,
            num_kv_heads,
            head_tiles,
            1, // read the first phase
            in_num_cores_x,
            in_num_cores_y,
            process_qv, //read and write q and v heads
            process_k  // read and write k heads
        };
        auto q_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
            q_cores,
            tt_metal::ReaderDataMovementConfig(q_reader_compile_time_args));
        std::vector<uint32_t> q_writer_compile_time_args = q_reader_compile_time_args;
        q_writer_compile_time_args[9] = 2;  // read the second phase
        auto q_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
            q_cores,
            tt_metal::WriterDataMovementConfig(q_writer_compile_time_args));

        KernelHandle k_reader_kernel_id = 0, k_writer_kernel_id = 0;
        if (!overlap_qk_coregrid) {
            // Switch process_qv and process_k for k kernels
            process_qv = 0;
            process_k = 1;
            std::vector<uint32_t> k_reader_compile_time_args = q_reader_compile_time_args;
            k_reader_compile_time_args[12] = process_qv;
            k_reader_compile_time_args[13] = process_k;
            k_reader_kernel_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
                k_cores,
                tt_metal::ReaderDataMovementConfig(k_reader_compile_time_args));

            std::vector<uint32_t> k_writer_compile_time_args = k_reader_compile_time_args;
            k_writer_compile_time_args[9] = 2; // read the second phase
            k_writer_kernel_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
                k_cores,
                tt_metal::WriterDataMovementConfig(k_writer_compile_time_args));
        }

    uint32_t q_start_addr = q_base_addr;
    const uint32_t device_batch_offset = batch_offset.value_or(0);

    for (uint32_t i = 0; i < q_num_cores; ++i) {
        uint32_t device_batch_idx = i + device_batch_offset;
        uint32_t in_tile_offset_by_batch = device_batch_idx < 16
                                               ? device_batch_idx * sub_tile_line_bytes
                                               : (device_batch_idx - 16) * sub_tile_line_bytes + 512 * element_size;

        const auto& core = q_cores_vector[i];
        std::vector<uint32_t> q_reader_runtime_args;
        q_reader_runtime_args.reserve(2 + in_num_cores_x + in_num_cores_y);
        q_reader_runtime_args = {
            in_tile_offset_by_batch,
            q_start_addr,
        };
        q_reader_runtime_args.insert(q_reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        q_reader_runtime_args.insert(q_reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        tt_metal::SetRuntimeArgs(program, q_reader_kernel_id, core, q_reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, q_writer_kernel_id, core, q_reader_runtime_args);
    }

        if (!overlap_qk_coregrid) {
            for (uint32_t i = 0; i < k_num_cores; ++i) {
                uint32_t device_batch_idx = i + device_batch_offset;
                uint32_t in_tile_offset_by_batch =
                    device_batch_idx < 16 ? device_batch_idx * sub_tile_line_bytes
                                          : (device_batch_idx - 16) * sub_tile_line_bytes + 512 * element_size;

                const auto& core = k_cores_vector[i];
                std::vector<uint32_t> k_reader_runtime_args;
                k_reader_runtime_args.reserve(2 + in_num_cores_x + in_num_cores_y);
                k_reader_runtime_args = {
                    in_tile_offset_by_batch,
                    q_start_addr,
                };
                k_reader_runtime_args.insert(k_reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
                k_reader_runtime_args.insert(k_reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

                tt_metal::SetRuntimeArgs(program, k_reader_kernel_id, core, k_reader_runtime_args);
                tt_metal::SetRuntimeArgs(program, k_writer_kernel_id, core, k_reader_runtime_args);
            }
        }

        auto override_runtime_arguments_callback =
            [q_reader_kernel_id,
             q_writer_kernel_id,
             k_reader_kernel_id,
             k_writer_kernel_id,
             q_num_cores,
             k_num_cores,
             cb_q_output,
             cb_k_output,
             cb_v_output,
             q_cores_vector,
             k_cores_vector,
             element_size,
             sub_tile_line_bytes,
             overlap_qk_coregrid,
             batch_offset,
             slice_size](
                const void* operation,
                Program& program,
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<Tensor>& output_tensors) {
                auto src_buffer = input_tensors.at(0).buffer();

                uint32_t src_kv_buffer_addr = 0;

                auto dst_buffer_query = output_tensors.at(0).buffer();
                auto dst_buffer_key = output_tensors.at(1).buffer();
                auto dst_buffer_value = output_tensors.at(2).buffer();

                UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
                UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
                UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

                uint32_t q_base_addr = input_tensors[0].buffer()->address();
                uint32_t q_start_addr = q_base_addr;

                uint32_t device_batch_offset = batch_offset.value_or(0);

                for (uint32_t i = 0; i < q_num_cores; ++i) {
                    uint32_t device_batch_idx = i + device_batch_offset;
                    uint32_t in_tile_offset_by_batch =
                        device_batch_idx < 16 ? device_batch_idx * sub_tile_line_bytes
                                              : (device_batch_idx - 16) * sub_tile_line_bytes + 512 * element_size;
                    const auto& core = q_cores_vector[i];
                    auto& runtime_args = GetRuntimeArgs(program, q_reader_kernel_id, core);
                    runtime_args[0] = in_tile_offset_by_batch;
                    runtime_args[1] = q_start_addr;

                    auto& runtime_args_writer = GetRuntimeArgs(program, q_writer_kernel_id, core);
                    runtime_args_writer[0] = in_tile_offset_by_batch;
                    runtime_args_writer[1] = q_start_addr;
                }

                if (!overlap_qk_coregrid) {
                    for (uint32_t i = 0; i < k_num_cores; ++i) {
                        uint32_t device_batch_idx = i + device_batch_offset;
                        uint32_t in_tile_offset_by_batch =
                            device_batch_idx < 16 ? device_batch_idx * sub_tile_line_bytes
                                                  : (device_batch_idx - 16) * sub_tile_line_bytes + 512 * element_size;
                        const auto& core = k_cores_vector[i];
                        auto& runtime_args = GetRuntimeArgs(program, k_reader_kernel_id, core);
                        runtime_args[0] = in_tile_offset_by_batch;
                        runtime_args[1] = q_start_addr;

                        auto& runtime_args_writer = GetRuntimeArgs(program, k_writer_kernel_id, core);
                        runtime_args_writer[0] = in_tile_offset_by_batch;
                        runtime_args_writer[1] = q_start_addr;
                    }
                }
            };

        return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace ttnn::operations::experimental::transformer
