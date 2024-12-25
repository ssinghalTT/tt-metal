// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_debug.h"
#include "compute_kernel_api.h"
#include "dprint.h"
#include "tensix_types.h"

// Given a Tensix configuration register field name, print the contents of the register.
// Uses tt_metal/hw/inc/<family>/cfg_defines.h:
//   For config section "Registers for THREAD", use banks THREAD_0_CFG, THREAD_1_CFG, THREAD_2_CFG
//   For other config sections (ALU,PACK0), use banks HW_CFG_0, HW_CFG_1
#define READ_CFG_REG_FIELD(bank, reg_field_name) \
    (dbg_read_cfgreg(bank, reg_field_name##_ADDR32) & reg_field_name##_MASK) >> reg_field_name##_SHAMT

// Helper macros
#define READ_HW_CFG_0_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_0, reg_field_name)
#define READ_HW_CFG_1_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_1, reg_field_name)
#define READ_THREAD_0_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_0_CFG, reg_field_name)
#define READ_THREAD_1_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_1_CFG, reg_field_name)
#define READ_THREAD_2_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_2_CFG, reg_field_name)

constexpr int PRECISION = 4;
constexpr int WIDTH = 8;

constexpr uint16_t NUM_FACES_PER_TILE = 4;
constexpr uint16_t NUM_ROWS_PER_FACE = 16;
constexpr uint16_t NUM_ROWS_PER_TILE = NUM_FACES_PER_TILE * NUM_ROWS_PER_FACE;

// Helper function to print array
inline void dprint_array_with_data_type(uint32_t data_format, uint32_t* data, uint32_t count) {
    DPRINT << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format, data, count)
           << ENDL();
}

// if flag DEST_ACCESS_CFG_remap_addrs is enabled
// destination register row identifiers are remmaped
// bits 5:3 are rotated 543 -> 354
inline uint16_t get_remapped_row_id(uint16_t row_id) {
    // bits 5:3 are rotating -> 543 -> 354
    return (row_id & 0xFFC7) |         // clear bits [5:3]
           ((row_id & 0x0008) << 2) |  // shifting bit 3 to position 5
           ((row_id & 0x0030) >> 1);   // shifting bits 5:4 to position 4:3
}

// if flag DEST_ACCESS_CFG_swizzle_32b is enabled dest address is has bits [3:2] shuffled
inline uint16_t get_swizzled_row_id(uint16_t row_id) {
    if (row_id & 0x10) {
        switch ((row_id & 0xC) >> 2) {
            case 0: return (row_id & 0xFFF3) | 0x8;
            case 1: return (row_id & 0xFFF3);
            case 2: return (row_id & 0xFFF3) | 0xC;
            case 3:
            default: return (row_id & 0xFFF3) | 0x4;
        }
    } else {
        return (row_id & 0xFFF3) | ((row_id & 0x4) << 1) | ((row_id & 0x8) >> 1);
    }
}

// Calculates dest row address based on logical row identifiers (tile_id, face_id, row_id)
// and dest configuration.
inline uint16_t get_dest_row_id(
    uint16_t tile_id, uint16_t face_id, uint16_t row_id, bool is_float32, bool is_remap, bool is_swizzle) {
    uint16_t row = NUM_ROWS_PER_TILE * tile_id + NUM_ROWS_PER_FACE * face_id + row_id;

    if (is_remap) {
        row = get_remapped_row_id(row);
    }

    if (is_float32) {
        if (is_swizzle) {
            row = get_swizzled_row_id(row);
        }
        // 0-7  dest rows for Float16
        // 8-15 dest rows for Mantissa
        // need to shift row index starting from bit 3
        row = ((row & 0xFFF8) << 1) | (row & 0x7);
    }

    return row;
}

inline uint16_t lo_word(uint32_t dword) { return dword & 0xFFFF; }
inline uint16_t hi_word(uint32_t dword) { return lo_word(dword >> 16); }

// Float16 = [1-bit sign, 7-bit mantissa, 8-bit exponent]
// Mantissa16 = [16-bit mantissa]
// Float32 = [1-bit sign, 8-bit exponent, 23-bit mantissa(7-bit + 16-bit)]
inline uint32_t reconstruct_float32(uint32_t float16, uint32_t mantissa16) {
    uint32_t sign = (float16 & 0x00008000) << 16;
    uint32_t exponent = (float16 & 0x000000FF) << 23;
    uint32_t mantissa = ((float16 & 0x00007F00) << 8) | mantissa16;

    return sign | exponent | mantissa;
}

// Helper function that prints one row from dest when dest is configured for storing float32 values.
// This function should be used only from dprint_tensix_dest_reg.
// Float32 in dest = [Float16, Mantissa16]
// dest_row -> [[Float16_1,Float16_0],...[Float16_15, Float16_14]]
// dest_row + 8 -> [[Mantissa16_1,Mantissa16_0],...[Mantissa16_15, Mantissa16_14]]
inline void dprint_tensix_dest_reg_row_float32(uint16_t row) {
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data_temp[ARRAY_LEN];
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type

    // read two rows [[Float16], [Mantissa]]
    dbg_read_dest_acc_row(row, rd_data_temp);
    dbg_read_dest_acc_row(row + 8, rd_data_temp + 8);

    for (int i = 0; i < 8; ++i) {
        rd_data[2 * i] = reconstruct_float32(lo_word(rd_data_temp[i]), lo_word(rd_data_temp[i + 8]));
        rd_data[2 * i + 1] = reconstruct_float32(hi_word(rd_data_temp[i]), hi_word(rd_data_temp[i + 8]));
    }

    dprint_array_with_data_type((uint32_t)DataFormat::Float32, rd_data, ARRAY_LEN);
}

// Helper function that prints one row from dest when dest is configured for storing float16 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_float16(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type(data_format, rd_data, 8);
}

// Print the contents of tile with index tile_id within the destination register
template <bool print_by_face = false>
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        // Determine the format of the data in the destination register
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);

#ifndef ARCH_GRAYSKULL
        // ALU_ACC_CTRL_Fp32 does not exist for GS
        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value =
                (uint32_t)DataFormat::Float32;  // Override the data format to tt::DataFormat::Float32
        }
#endif

        bool is_float32 = data_format_reg_field_value == (uint32_t)DataFormat::Float32;
        bool is_swizzled = false;
        bool is_remapped = false;

#ifdef ARCH_BLACKHOLE
        is_remapped = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_remap_addrs) == 1;
        is_swizzled = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_swizzle_32b) == 1;
#endif
        // Print the contents
        DPRINT << FIXED() << SETW(WIDTH) << SETPRECISION(PRECISION);
        DPRINT << "Tile ID = " << tile_id << ENDL();

        for (int face_id = 0; face_id < NUM_FACES_PER_TILE; ++face_id) {
            for (int row_id = 0; row_id < NUM_ROWS_PER_FACE; ++row_id) {
                uint16_t row = get_dest_row_id(tile_id, face_id, row_id, is_float32, is_remapped, is_swizzled);
                if (is_float32) {
                    dprint_tensix_dest_reg_row_float32(row);
                } else {
                    dprint_tensix_dest_reg_row_float16(data_format_reg_field_value, row);
                }
            }
            if constexpr (print_by_face) {
                DPRINT << ENDL();
            }
        }
    })
    dbg_unhalt();
}

// Print the contents of the specified configuration register field.
// Example:
//   dprint_cfg_reg_field(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg_field(bank, reg_field_name)                                          \
    {                                                                                       \
        uint32_t field_val = READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::bank, reg_field_name); \
        DPRINT << #reg_field_name << " = " << field_val << ENDL();                          \
    }

// Print the contents of the whole configuration register. The register is specified by
// the name of any field within it.
// Example:
//    dprint_cfg_reg(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg(bank, reg_field_name)                                                    \
    {                                                                                           \
        uint32_t reg_val = dbg_read_cfgreg(ckernel::dbg_cfgreg::bank, reg_field_name##_ADDR32); \
        DPRINT << #reg_field_name << " = " << HEX() << reg_val << ENDL();                       \
    }

// Print the content of the register field given the value in the register.
#define DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, reg_field_name, name)                           \
    {                                                                                           \
        uint32_t field_value = (reg_val & reg_field_name##_MASK) >> reg_field_name##_SHAMT;     \
        DPRINT << name << " = " << HEX() << field_value << "; ";                                \
    }

// Print content of the register field by field. Issue: No ENDL.
inline void dprint_tensix_alu_config(){
        uint32_t reg_val = dbg_read_cfgreg(ckernel::dbg_cfgreg::HW_CFG_0, ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32);
        DPRINT << "RND_MODE: ";                                                           
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Fpu_srnd_en, "Fpu_srnd_en");          
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Gasket_srnd_en, "Gasket_srnd_en");    
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Packer_srnd_en, "Packer_srnd_en");    
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Padding, "Padding");                  
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_GS_LF, "GS_LF");                      
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Bfp8_HF, "Bfp8_HF");
        DPRINT << "FORMAT: ";                     
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcAUnsigned, "SrcAUnsigned");     
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcBUnsigned, "SrcBUnsigned");     
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcA, "SrcA");                     
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG1_SrcB, "SrcB");                     
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG2_Dstacc, "Dstacc");
        DPRINT << "ACC_CTRL: ";                      
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_Fp32_enabled, "Fp32_enabled");             
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_SFPU_Fp32_enabled, "SFPU_Fp32_enabled");   
        DPRINT_TENSIX_ALU_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_INT8_math_enabled, "INT8_math_enabled");   
        DPRINT << ENDL();  
}

inline void dprint_tensix_unpack_tile_descriptor(){
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //TEST
    /*cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = 0xfffffff5;
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1] = 0xffffffff;
    cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32] = 0x5678ffff;
    cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1] = 0xffff1234;
    */

    //word 0
    uint32_t word0 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32];
    DPRINT << HEX() << word0 << "; "; 
    DPRINT << HEX() << ((word0 & 0xf) >> 0) << "; "; // in_data_format
    DPRINT << HEX() << ((word0 & 0x10) >> 4) << "; "; // uncompressed
    DPRINT << HEX() << ((word0 & 0xe0) >> 5) << "; "; // reserved_0
    DPRINT << HEX() << ((word0 & 0xf00) >> 8) << "; "; // blobs_per_xy_plane
    DPRINT << HEX() << ((word0 & 0xf000) >> 12) << "; "; // reserved_1
    DPRINT << HEX() << ((word0 & 0xffff0000) >> 16) << "; "; // x_dim

    //word 1
    uint32_t word1 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
    DPRINT << HEX() << word1 << "; ";
    DPRINT << HEX() << ((word1 & 0xffff) >> 0) << "; "; // y_dim
    DPRINT << HEX() << ((word1 & 0xffff0000) >> 16) << "; "; //z_dim

    //word 2
    uint32_t word2 = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32];
    DPRINT << HEX() << word2 << "; ";
    DPRINT << HEX() << ((word2 & 0xffff) >> 0) << "; "; //w_dim
    
    // blobs_y_start is in 2 words (word2 and word3)
    // word3
    uint32_t word3 = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1];
    DPRINT << HEX() << word3 << "; ";
    DPRINT << HEX() << (((word3 & 0xffff) << 16) | ((word2 & 0xffff0000) >> 16)) << "; "; //blobs_y_start
    DPRINT << HEX() << ((word3 & 0xff0000) >> 16) << "; "; //digest_type
    DPRINT << HEX() << ((word3 & 0xff000000) >> 24) << "; "; //digest_size 

    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config(){
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //TEST
    /*
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = 0x00000025;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 1] = 0x000f000f;
    cfg[THCON_SEC1_REG2_Out_data_format_ADDR32] = 0x00000025;
    cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + 1] = 0x000f000f;
    */

    //word 0
    uint32_t word0 = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32];
    DPRINT << "w0: " << HEX() << word0 << "; "; 
    DPRINT << HEX() << ((word0 & 0xf) >> 0) << "; "; //out_data_format
    DPRINT << HEX() << ((word0 & 0x30) >> 4) << "; "; //throttle_mode
    DPRINT << HEX() << ((word0 & 0xc0) >> 6) << "; "; //context_count
    DPRINT << HEX() << ((word0 & 0x100) >> 8) << "; "; //haloize_mode
    DPRINT << HEX() << ((word0 & 0x200) >> 9) << "; "; //tileize_mode
    DPRINT << HEX() << ((word0 & 0x400) >> 10) << "; "; //force_shared_exp
    DPRINT << HEX() << ((word0 & 0x800) >> 11) << "; "; //reserved_0
    DPRINT << HEX() << ((word0 & 0x7000) >> 12) << "; "; //upsample_rate
    DPRINT << HEX() << ((word0 & 0x8000) >> 15) << "; "; //upsample_and_interlave
    DPRINT << HEX() << ((word0 & 0xffff0000) >> 16) << "; "; //shift_amount

    //word 2
    uint32_t word1 = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 1];
    DPRINT << "w1: " << HEX() << word1 << "; ";
    DPRINT << HEX() << ((word1 & 0xf) >> 0) << "; "; //uncompress_cntx0_3
    DPRINT << HEX() << ((word1 & 0xfff0) >> 4) << "; "; //reserved_1
    DPRINT << HEX() << ((word1 & 0xf0000) >> 16) << "; "; //uncompress_cntx4_7
    DPRINT << HEX() << ((word1 & 0xfff00000) >> 20) << "; "; //reserved_2

    //word 2
    uint32_t word2 = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32];
    DPRINT << "w2: " << HEX() << word2 << "; ";
    DPRINT << HEX() << ((word2 & 0xffff) >> 0) << "; "; //limit_addr
    DPRINT << HEX() << ((word2 & 0xffff0000) >> 16) << "; "; //fifo_size

    //word 3
    uint32_t word3 = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + 1];
    DPRINT << "w3: " << HEX() << word3 << "; ";

    DPRINT << ENDL();
}