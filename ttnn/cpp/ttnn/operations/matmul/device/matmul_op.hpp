// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <sstream>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/types.hpp"

namespace tt {

namespace tt_metal {

using ttnn::operations::unary::UnaryWithParam;

/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool mcast_in0, bool untilize_out);
operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t per_core_M, uint32_t per_core_N, std::optional<UnaryWithParam> fused_activation, bool untilize_out, bool skip_compute, bool skip_in0_mcast, bool skip_write_back);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool transpose_mcast, std::optional<UnaryWithParam> fused_activation, bool untilize_out);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor &output_tensor, bool bcast_batch, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool untilize_out);


/**
 * Generalized blocked matmul with support for tilize and untilize and mixed-prec
 */
struct BMMTilizeUntilize {
    const DataType out_dt_;
    const uint32_t in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_;
    const uint32_t in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_;
    const uint32_t out_subblock_ntiles_h_, out_subblock_ntiles_w_;
    const bool tilize_in0_, untilize_out_;
    const bool has_bias_;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

/**
 * Blocked Matmul, with support for tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b, const Tensor& bias, DataType out_dt,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_in0, bool untilize_out, bool has_bias,
                           std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
                                    const Tensor &in0, const Tensor &in1, Tensor &bias, DataType out_dt,
                                    uint32_t in0_height_nblocks, uint32_t in0_width_nblocks, uint32_t in1_width_nblocks,
                                    uint32_t in0_block_height_ntiles, uint32_t in0_block_width_ntiles, uint32_t in1_block_width_ntiles,
                                    uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                                    bool tilize_in0, bool untilize_out, bool has_bias,
                                    Tensor &out, DeviceComputeKernelConfig compute_kernel_config);

}  // namespace tt_metal


namespace operations {

namespace primary {

using namespace tt_metal;

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {
};

struct MatmulMultiCoreNonOptimizedReuseProgramConfig {
};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreNonOptimizedReuseProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
>;

struct Matmul {
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    const std::optional<bool> bcast_batch = std::nullopt;
    const MemoryConfig output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    const std::optional<DataType> output_dtype = std::nullopt;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    const bool untilize_out = false;
    const std::optional<const CoreCoord> user_core_coord = std::nullopt;
    const std::optional<UnaryWithParam> user_fused_activation = std::nullopt;
    const bool user_run_batched = false;
    const bool transpose_a = false;
    const bool transpose_b = false;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes_dram_sharded(const std::vector<Tensor>& input_tensors, uint32_t N_unpadded) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

inline bool get_broadcast_batch(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const MatmulProgramConfig> matmul_program_config) {
    uint32_t batch_size_b = get_batch_size(input_tensor_b.get_legacy_shape());
    bool broadcast_batch = batch_size_b == 1;
    if (!matmul_program_config.has_value()) {
        return broadcast_batch;
    }

    bool is_multi_core_reuse = std::visit(
        [](const auto& program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                return true;
            }
            return false;
        },
        matmul_program_config.value()
    );
    if (is_multi_core_reuse) {
        uint32_t batch_size_a = get_batch_size(input_tensor_a.get_legacy_shape());
        broadcast_batch &= batch_size_a > 1;
    }
    return broadcast_batch;
}

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(const ttnn::types::Shape& input_shape_a, const ttnn::types::Shape& input_shape_b, const CoreCoord& core_coord, const std::optional<UnaryWithParam> fused_activation, const bool fp32_dest_acc_en);

MatmulProgramConfig create_matmul_program_config(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<const CoreCoord> user_core_coord, std::optional<UnaryWithParam> fused_activation, std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

const struct Matmul generate_matmul_struct(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const struct Matmul& parameters);

std::string get_matmul_validate_string(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> optional_bias,
    const struct Matmul& parameters);

uint32_t get_matmul_cbs_size_in_bytes(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> optional_bias,
    const struct Matmul& parameters);

uint32_t get_matmul_output_tensor_size_in_bytes(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> optional_bias,
    const struct Matmul& parameters);

std::stringstream get_matmul_validate_stream(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> optional_bias,
    const struct Matmul& matmul);

inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias = std::nullopt,
    const struct Matmul& parameters = Matmul{}) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}, {bias.value()}))};
    } else {
        optional_input_tensors.push_back(std::nullopt);
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
    }
    const struct Matmul& matmul = generate_matmul_struct(input_tensor_a, input_tensor_b, parameters);

    operation::launch_op(
            [matmul] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
        const auto& input_tensor_a = input_tensors.at(0);
        const auto& input_tensor_b = input_tensors.at(1);
        return operation::run(matmul, {input_tensor_a, input_tensor_b}, optional_input_tensors);
    },
    {input_tensor_a, input_tensor_b}, output_tensors, optional_input_tensors);
    return output_tensors.at(0);
}

MatmulProgramConfig generate_matmul_program_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig &mem_config, const std::optional<const DeviceComputeKernelConfig> compute_kernel_config, const std::optional<const CoreCoord> user_core_coord, const std::optional<UnaryWithParam> user_fused_activation, const std::optional<const bool> user_run_batched);

}  // namespace primary

}  // namespace operations

}  // namespace tt


namespace bmm_op_utils {
using namespace tt::tt_metal;

// Ensure there are always symmetrical values. Different paths use different
// index ordering (0,1 vs 1,0) to meet test PCC requirements.
constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},  // subblock_hw = 8
    {7, 1}, {1, 7},                  // subblock_hw = 7
    {3, 2}, {2, 3}, {6, 1}, {1, 6},  // subblock_hw = 6
    {5, 1}, {1, 5},                  // subblock_hw = 5
    {2, 2}, {4, 1}, {1, 4},          // subblock_hw = 4
    {3, 1}, {1, 3},                  // subblock_hw = 3
    {2, 1}, {1, 2},                  // subblock_hw = 2
    {1, 1},                          // subblock_hw = 1
}};

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

inline bool get_fp32_dest_acc_en(const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    bool fp32_dest_acc_en = false;
    if (compute_kernel_config) {
        std::visit([&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            }
        }, *compute_kernel_config);
    }
    return fp32_dest_acc_en;
}

// TODO: Remove get_mcast_1d_config and merge with general version?
tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool fuse_batch = false, std::optional<UnaryWithParam> fused_activation = std::nullopt, bool mcast_in0 = true, bool out_sharded = false, std::optional<CoreCoord> compute_with_storage_grid_size = std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

tuple<uint32_t, uint32_t> get_matmul_subblock_params(const uint32_t per_core_M, const uint32_t per_core_N, const bool per_core_M_equals_subblock_h_constraint, bool per_core_N_equals_subblock_w_constraint, bool fp32_dest_acc_en);

// TODO: Review usage of matmul bool; should probably infer this from batch
tt::operations::primary::MatmulProgramConfig get_matmul_program_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig &output_mem_config, std::optional<UnaryWithParam> fused_activation = std::nullopt, const bool matmul = false, const std::optional<const CoreCoord> user_core_coord = std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}  // namespace bmm_op_utils
