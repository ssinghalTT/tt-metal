// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

#ifndef SKIP_UNP
    #define SKIP_UNP (0)
#endif

inline void llk_unpack_untilize_mop_config() {

    constexpr uint replay_buf_len = 5;
    TTI_REPLAY(0, replay_buf_len, 0, 1);

    TTI_DMANOP; // REG2FLOP that sets offset in previous loop needs additional cycle to complete
    TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
    TTI_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001);
    static constexpr uint load_offset_addr_cntx0 = TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);
    static constexpr uint load_offset_addr_cntx1 = TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);

    ckernel_unpack_template tmp = ckernel_unpack_template(
          true,  // src B
          false,  // halo - just used for 4 unpacks
          TT_OP_REPLAY(0, replay_buf_len, 0, 0),
          0,
          0,
          0,
          TT_OP_REPLAY(0, replay_buf_len, 0, 0),
          load_offset_addr_cntx0,
          load_offset_addr_cntx1);
    tmp.program(instrn_buffer);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_untilize_hw_configure(const llk_unpack_A_params_t *unpack_untilize_params) {
    constexpr bool is_row_pool = false;
    constexpr bool transpose_xy_srca = false;
    constexpr bool srnd_fpu_en = false;

    const uint32_t unpA_operand_id = get_operand_id(unpack_untilize_params->unpA_operand);
    const uint32_t unpA_num_faces = 4;
    const uint32_t unpA_face_r_dim = 16;
    configure_unpack_AB(unpA_operand_id, unpA_operand_id, unpA_face_r_dim, unpA_face_r_dim, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en, srnd_fpu_en, unpA_num_faces, unpA_num_faces);
}

inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure(&unpack_untilize_params);
}

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t face_r_dim = 1;

    std::uint32_t unpA_ch1_x_stride = (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    std::uint32_t unpA_ch1_y_stride = FACE_R_DIM*unpA_ch1_x_stride;

    TT_SETADCXX(p_setadc::UNP_A, face_r_dim*FACE_C_DIM-1, 0x0);

    // Get pointer to registers for current state ID
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 0, 0xFFFF>(FACE_C_DIM);
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

    std::uint32_t tile_size_words = cb_interface[operand_id].fifo_page_size;
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size_words), 0, LO_16(p_gpr_unpack::TILE_SIZE));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_size_words), 0, HI_16(p_gpr_unpack::TILE_SIZE));
    llk_unpack_untilize_mop_config();
}

inline void llk_unpack_untilize_uninit(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t unpA_ch1_x_stride = (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[operand_id]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    std::uint32_t unpA_ch1_y_stride = FACE_C_DIM*FACE_R_DIM*unpA_ch1_x_stride;

    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    // Wait for cfg to be free to edit
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    // Reset the values to default in unpack AB common.
    TT_SETADCXX(p_setadc::UNP_A, FACE_R_DIM*FACE_C_DIM-1, 0x0);
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 0, 0xFFFF>(1);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(unpA_ch1_y_stride);
    TTI_NOP; TTI_NOP; // Do we need this for WH?
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;
    std::uint32_t rem_blocks_in_row = block_tile_cols;

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b0010);  // Clear l1 addr y cnt
    if constexpr (first_pass) {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 0);
    } else {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 2);
    }

    // Wait for free context
    wait_for_next_context(2);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    if (0 == unp_cfg_context) {
       cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base_address;
    } else {
       cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = base_address;
    }

    std::uint32_t face_2xr_cnt = 0;
    for (std::uint32_t r = 0; r < FACE_HEIGHT; r++) {
        rem_blocks_in_row = block_tile_cols;  // reset remaining blocks in row

        do {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2)) {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, unp_cfg_context == 0 ? 0 : 0xff);                                              // Run the MOP

                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // set data valid
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);

                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            } else {
                TT_MOP(0, rem_blocks_in_row - 1, unp_cfg_context == 0 ? 0 : 0xff);  // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
                // if (face_2xr_cnt==FACE_HEIGHT/2) {
                //   TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); //set data valid
                //   TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                //   face_2xr_cnt = 0;
                //}
            }
        } while (rem_blocks_in_row > 0);

        TTI_MULDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr::ZERO); // TILE_OFFSET=TILE_OFFSET*0
        if (0 == unp_cfg_context) {
            TTI_REG2FLOP(
                1,
                0,
                0,
                0,
                THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
                p_gpr::ZERO);                 // Clear offset register
        } else {
            TTI_REG2FLOP(
                1,
                0,
                0,
                0,
                THCON_SEC0_REG7_Offset_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
                p_gpr::ZERO);                 // Clear offset register
        }
        TTI_INCADCXY(0b001, 0, 0, 1, 0);  // inc l1 addr y cnt
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
}
