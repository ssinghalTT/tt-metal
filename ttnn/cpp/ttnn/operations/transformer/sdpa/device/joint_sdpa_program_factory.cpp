// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "joint_sdpa_program_factory.hpp"
#include "joint_sdpa_op.hpp"

#include <optional>
#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::transformer::detail {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks joint_sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& joint_tensor_q,
    const Tensor& joint_tensor_k,
    const Tensor& joint_tensor_v,
    const Tensor& output_tensor,
    const Tensor& joint_output_tensor,
    std::optional<float> scale,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config) {
    /*
    Q: B x NH x N x DH
    K: B x NH x N x DH
    V: B x NH x N x DH

    Q_joint: B x NH x L x DH
    K_joint: B x NH x L x DH
    V_joint: B x NH x L x DH
    */

    const auto q_shape = input_tensor_q.get_logical_shape();
    const auto joint_q_shape = joint_tensor_q.get_logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], N = q_shape[2], DH = q_shape[3];
    const uint32_t L = joint_q_shape[2];

    // Calculate padded sequence length
    const uint32_t padded_Nq = tt::round_up(N, q_chunk_size);
    const uint32_t padded_Nk = tt::round_up(N, k_chunk_size);
    const uint32_t padded_Lq = tt::round_up(L, q_chunk_size);
    const uint32_t padded_Lk = tt::round_up(L, k_chunk_size);

    const uint32_t padded_Nqt = padded_Nq / TILE_HEIGHT;
    const uint32_t padded_Nkt = padded_Nk / TILE_HEIGHT;
    const uint32_t padded_Lqt = padded_Lq / TILE_HEIGHT;
    const uint32_t padded_Lkt = padded_Lk / TILE_HEIGHT;

    // Find unpadded sequence lengths in tiles
    const uint32_t valid_Nt = tt::div_up(N, TILE_HEIGHT);
    const uint32_t valid_Lt = tt::div_up(L, TILE_HEIGHT);

    // Compute kernel operates on concatenated Q and K
    const uint32_t cat_Sq = padded_Nq + padded_Lq;
    const uint32_t cat_Sk = padded_Nk + padded_Lk;

    const uint32_t cat_Sqt = cat_Sq / TILE_HEIGHT;
    const uint32_t cat_Skt = cat_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;

    // Kernel will need to know the tile-based shapes of both sets of tensors
    // to create a representation of the concatenated tensors.

    // const std::vector<uint32_t> q_tile_shape = {B, NH, padded_Nqt, DHt};
    // const std::vector<uint32_t> k_tile_shape = {B, NH, padded_Nkt, DHt};
    // const std::vector<uint32_t> joint_q_tile_shape = {B, NH, padded_Lqt, DHt};
    // const std::vector<uint32_t> joint_k_tile_shape = {B, NH, padded_Lkt, DHt};

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */
    const bool use_joint_mask = (padded_Nk != N) || (padded_Lk != L);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = cat_Sq / q_chunk_size;
    const uint32_t k_num_chunks = cat_Sk / k_chunk_size;

    tt::log_debug("B: {}", B);
    tt::log_debug("NH: {}", NH);
    tt::log_debug("N: {}", N);
    tt::log_debug("L: {}", L);
    tt::log_debug("DH: {}", DH);

    // Log padded dimensions
    tt::log_debug("padded_Nq: {}", padded_Nq);
    tt::log_debug("padded_Nk: {}", padded_Nk);
    tt::log_debug("padded_Lq: {}", padded_Lq);
    tt::log_debug("padded_Lk: {}", padded_Lk);
    tt::log_debug("padded_Nqt: {}", padded_Nqt);
    tt::log_debug("padded_Nkt: {}", padded_Nkt);
    tt::log_debug("padded_Lqt: {}", padded_Lqt);
    tt::log_debug("padded_Lkt: {}", padded_Lkt);

    // Log tile dimensions
    tt::log_debug("DHt: {}", DHt);
    tt::log_debug("valid_Nt: {}", valid_Nt);
    tt::log_debug("valid_Lt: {}", valid_Lt);

    // Log chunking parameters
    tt::log_debug("Sq_chunk_t: {}", Sq_chunk_t);
    tt::log_debug("Sk_chunk_t: {}", Sk_chunk_t);
    tt::log_debug("q_chunk_size: {}", q_chunk_size);
    tt::log_debug("k_chunk_size: {}", k_chunk_size);
    tt::log_debug("q_num_chunks: {}", q_num_chunks);
    tt::log_debug("k_num_chunks: {}", k_num_chunks);

    // Log concatenated dimensions
    tt::log_debug("cat_Sq: {}", cat_Sq);
    tt::log_debug("cat_Sk: {}", cat_Sk);
    tt::log_debug("cat_Sqt: {}", cat_Sqt);
    tt::log_debug("cat_Skt: {}", cat_Skt);

    tt::log_debug("use_joint_mask: {}", use_joint_mask);

    Program program = CreateProgram();

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        program_config.has_value()
            ? (program_config->exp_approx_mode.has_value() ? program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);

    TT_FATAL(
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor,
        num_cores);

    tt::log_debug("Parallelization scheme:");
    tt::log_debug("batch_parallel_factor: {}", batch_parallel_factor);
    tt::log_debug("nh_parallel_factor: {}", nh_parallel_factor);
    tt::log_debug("q_parallel_factor: {}", q_parallel_factor);

    // Ceiling divide to allow for non-perfect divisions
    const uint32_t batch_per_core = tt::div_up(B, batch_parallel_factor);
    const uint32_t nh_per_core = tt::div_up(NH, nh_parallel_factor);
    const uint32_t q_per_core = tt::div_up(q_num_chunks, q_parallel_factor);

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;

    tt::log_debug("q_per_core: {}", q_per_core);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    tt::log_debug("q_tiles: {}", q_tiles);
    tt::log_debug("k_tiles: {}", k_tiles);
    tt::log_debug("v_tiles: {}", v_tiles);
    tt::log_debug("mask_tiles: {}", mask_tiles);
    tt::log_debug("qk_tiles: {}", qk_tiles);
    tt::log_debug("out0_t: {}", out0_t);
    tt::log_debug("scale_tiles: {}", scale_tiles);
    tt::log_debug("statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    const uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    const uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(DHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == DHt) ? (std::min(Sq_chunk_t, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // log all values
    tt::log_debug("dst_size: {}", dst_size);
    tt::log_debug("qk_in0_block_w: {}", qk_in0_block_w);
    tt::log_debug("qk_out_subblock_w: {}", qk_out_subblock_w);
    tt::log_debug("qk_out_subblock_h: {}", qk_out_subblock_h);
    tt::log_debug("qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    tt::log_debug("qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    tt::log_debug("qk_num_blocks: {}", qk_num_blocks);
    tt::log_debug("out_in0_block_w: {}", out_in0_block_w);
    tt::log_debug("out_out_subblock_w: {}", out_out_subblock_w);
    tt::log_debug("out_out_subblock_h: {}", out_out_subblock_h);
    tt::log_debug("out_in0_num_subblocks: {}", out_in0_num_subblocks);
    tt::log_debug("out_in1_num_subblocks: {}", out_in1_num_subblocks);
    tt::log_debug("out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    const uint32_t stats_granularity = std::min(Sq_chunk_t, dst_size);
    // Find log2 of stats_granularity using std
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    // Assert that this is a power of 2
    TT_FATAL(
        stats_granularity == (1 << log2_stats_granularity),
        "stats_granularity must be a power of 2. Got {}.",
        stats_granularity);

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    const uint32_t mul_bcast_granularity = std::min(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
    TT_FATAL(
        mul_bcast_granularity == (1 << log2_mul_bcast_granularity),
        "mul_bcast_granularity must be a power of 2. Got {}.",
        mul_bcast_granularity);

    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = std::log2(dht_granularity);
    // Sometimes DHt is not a power of 2, so granularity should be 1
    if (dht_granularity != (1 << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }
    TT_FATAL(
        dht_granularity == (1 << log2_dht_granularity),
        "dht_granularity must be a power of 2. Got {}.",
        dht_granularity);

    // Log these
    tt::log_debug("stats_granularity: {}", stats_granularity);
    tt::log_debug("log2_stats_granularity: {}", log2_stats_granularity);
    tt::log_debug("sub_exp_granularity: {}", sub_exp_granularity);
    tt::log_debug("log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    tt::log_debug("mul_bcast_granularity: {}", mul_bcast_granularity);
    tt::log_debug("log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    tt::log_debug("dht_granularity: {}", dht_granularity);
    tt::log_debug("log2_dht_granularity: {}", log2_dht_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union;
    scale_union.f = scale.value_or(1.0f);

    // log scale
    tt::log_debug("scale: {}", scale_union.f);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        valid_Nt,
        valid_Lt,
        padded_Nqt,
        padded_Nkt,
        padded_Lqt,
        padded_Lkt,
        num_cores,
    };

    // Calculate which K chunks contain the mask boundaries
    // If a tensor does not require masking, set to MAX_UINT32. This avoids a
    // bug in the mask generation code, which would mask a full, valid chunk
    // with -inf.
    const uint32_t mask_chunk_0 =
        (padded_Nk != N) ? (padded_Nkt / Sk_chunk_t) - 1 : (uint32_t)(-1);  // idx of last chunk in first sequence
    const uint32_t mask_chunk_1 =
        (padded_Lk != L) ? (cat_Skt / Sk_chunk_t) - 1 : (uint32_t)(-1);  // idx of last chunk in second sequence

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        valid_Nt,
        valid_Lt,
        padded_Nqt,
        padded_Nkt,
        padded_Lqt,
        padded_Lkt,
        N,
        L,
        num_cores,
        packed_identity_scalar,
        scale_union.u,
        (uint32_t)use_joint_mask,
        mask_chunk_0,
        mask_chunk_1,
    };

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        cat_Skt,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        (uint32_t)use_joint_mask,
        mask_chunk_0,
        mask_chunk_1,
    };

    std::map<string, string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/joint_sdpa.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.get_dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.get_dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.get_dtype());
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tt_metal::detail::TileSize(q_df);
    uint32_t k_tile_size = tt::tt_metal::detail::TileSize(k_df);
    uint32_t v_tile_size = tt::tt_metal::detail::TileSize(v_df);
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_df);
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_df);
    uint32_t stats_tile_size = tt::tt_metal::detail::TileSize(stats_df);

    log_debug("q_data_format: {}", q_df);
    log_debug("k_data_format: {}", k_df);
    log_debug("v_data_format: {}", v_df);
    log_debug("mask_data_format: {}", mask_df);
    log_debug("out_data_format: {}", out_df);
    log_debug("scalar_data_format: {}", scalar_df);
    log_debug("intermediate_data_format: {}", im_df);
    log_debug("statistics_data_format: {}", stats_df);

    // Q input
    auto c_in0_config = CircularBufferConfig(q_tiles * q_tile_size, {{tt::CBIndex::c_0, q_df}})
                            .set_page_size(tt::CBIndex::c_0, q_tile_size);

    auto cb_in0_id = CreateCircularBuffer(program, core_grid, c_in0_config);
    // K input
    auto c_in1_config = CircularBufferConfig(k_tiles * k_tile_size, {{tt::CBIndex::c_1, k_df}})
                            .set_page_size(tt::CBIndex::c_1, k_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, core_grid, c_in1_config);
    // V input
    auto c_in2_config = CircularBufferConfig(v_tiles * v_tile_size, {{tt::CBIndex::c_2, v_df}})
                            .set_page_size(tt::CBIndex::c_2, v_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, core_grid, c_in2_config);

    // Only create mask buffer if it's going to be used
    if (use_joint_mask) {
        // attn_mask input
        auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
                                .set_page_size(tt::CB::c_in3, mask_tile_size);
        auto cb_in3_id = CreateCircularBuffer(program, core_grid, c_in3_config);
    }

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_4, scalar_df}})
                            .set_page_size(tt::CBIndex::c_4, scalar_tile_size);
    auto cb_in4_id = CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    auto cb_in5_id = CreateCircularBuffer(program, core_grid, c_in5_config);

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{tt::CBIndex::c_24, im_df}})
                                  .set_page_size(tt::CBIndex::c_24, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_25, im_df}})
                                  .set_page_size(tt::CBIndex::c_25, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_26, im_df}})
                                  .set_page_size(tt::CBIndex::c_26, im_tile_size);
    auto cb_intermed2_id = CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_27, stats_df}})
                                  .set_page_size(tt::CBIndex::c_27, stats_tile_size);
    auto cb_intermed3_id = CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_28, stats_df}})
                                  .set_page_size(tt::CBIndex::c_28, stats_tile_size);
    auto cb_intermed4_id = CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_29, stats_df}})
                                  .set_page_size(tt::CBIndex::c_29, stats_tile_size);
    auto cb_intermed5_id = CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_30, stats_df}})
                                  .set_page_size(tt::CBIndex::c_30, stats_tile_size);
    auto cb_intermed6_id = CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_31, stats_df}})
                                  .set_page_size(tt::CBIndex::c_31, stats_tile_size);
    auto cb_intermed7_id = CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CBIndex::c_16, out_df}})
                             .set_page_size(tt::CBIndex::c_16, out_tile_size);
    auto cb_out0_id = CreateCircularBuffer(program, core_grid, c_out0_config);

    uint32_t q_addr = input_tensor_q.buffer()->address();
    uint32_t k_addr = input_tensor_k.buffer()->address();
    uint32_t v_addr = input_tensor_v.buffer()->address();
    uint32_t joint_q_addr = joint_tensor_q.buffer()->address();
    uint32_t joint_k_addr = joint_tensor_k.buffer()->address();
    uint32_t joint_v_addr = joint_tensor_v.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t joint_out_addr = joint_output_tensor.buffer()->address();

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NH);
        local_nh_end = std::min(local_nh_end, NH);
        local_q_start = std::min(local_q_start, q_num_chunks);
        local_q_end = std::min(local_q_end, q_num_chunks);

        // log the above
        tt::log_debug("core: {}", i);
        tt::log_debug("x={},y={}", core.x, core.y);
        tt::log_debug("local_batch_start: {}", local_batch_start);
        tt::log_debug("local_batch_end: {}", local_batch_end);
        tt::log_debug("local_nh_start: {}", local_nh_start);
        tt::log_debug("local_nh_end: {}", local_nh_end);
        tt::log_debug("local_q_start: {}", local_q_start);
        tt::log_debug("local_q_end: {}", local_q_end);

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {q_addr,
             k_addr,
             v_addr,
             joint_q_addr,
             joint_k_addr,
             joint_v_addr,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end});

        // Writer args
        SetRuntimeArgs(
            program,
            writer_kernels_id,
            core,
            {out_addr,
             joint_out_addr,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end});

        // Compute args
        SetRuntimeArgs(
            program,
            compute_kernels_id,
            core,
            {local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end});
    }

    auto override_runtime_arguments_callback =
        [num_cores, grid_size, reader_kernels_id, writer_kernels_id, compute_kernels_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            // Get addresses for regular tensors
            auto q_buffer = input_tensors.at(0).buffer();
            auto k_buffer = input_tensors.at(1).buffer();
            auto v_buffer = input_tensors.at(2).buffer();
            auto joint_q_buffer = input_tensors.at(3).buffer();
            auto joint_k_buffer = input_tensors.at(4).buffer();
            auto joint_v_buffer = input_tensors.at(5).buffer();

            // Get addresses for output tensors
            auto out_buffer = output_tensors.at(0).buffer();
            auto joint_out_buffer = output_tensors.at(1).buffer();

            uint32_t q_addr = q_buffer->address();
            uint32_t k_addr = k_buffer->address();
            uint32_t v_addr = v_buffer->address();
            uint32_t joint_q_addr = joint_q_buffer->address();
            uint32_t joint_k_addr = joint_k_buffer->address();
            uint32_t joint_v_addr = joint_v_buffer->address();
            uint32_t out_addr = out_buffer->address();
            uint32_t joint_out_addr = joint_out_buffer->address();

            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
            auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};

                auto& reader_args = reader_args_by_core[core.x][core.y];
                auto& writer_args = writer_args_by_core[core.x][core.y];
                auto& compute_args = compute_args_by_core[core.x][core.y];

                // Update reader args
                reader_args[0] = q_addr;
                reader_args[1] = k_addr;
                reader_args[2] = v_addr;
                reader_args[3] = joint_q_addr;
                reader_args[4] = joint_k_addr;
                reader_args[5] = joint_v_addr;

                // Update writer args
                writer_args[0] = out_addr;
                writer_args[1] = joint_out_addr;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::transformer::detail
