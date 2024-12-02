// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with no operand
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t transpose = 0, const std::uint32_t acc_to_dest = 0) {
    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>(
        transpose, acc_to_dest);
}

// Version with operands
// TODO: is this needed? operands arent used for anything
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    const std::uint32_t transpose = 0,
    const std::uint32_t acc_to_dest = 0) {
    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>(
        transpose, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = false) {
    const std::uint32_t num_faces = 4;

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        NUM_FIDELITY_PHASES,
        binary_reuse_dest,
        is_fp32_dest_acc_en>(num_faces, num_faces, dst_index, clear_fp32_dst_acc);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    uint dst_index,
    const bool clear_fp32_dst_acc = false) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        NUM_FIDELITY_PHASES,
        binary_reuse_dest,
        is_fp32_dest_acc_en>(num_faces, num_faces, dst_index, clear_fp32_dst_acc);
}
