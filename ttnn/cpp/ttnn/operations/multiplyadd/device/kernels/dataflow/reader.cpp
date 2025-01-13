#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({ DPRINT << r << " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint8_t src1_cb_index = tt::CBIndex::c_1;
    uint8_t src2_cb_index = tt::CBIndex::c_2;
    uint32_t src0Addr = get_arg_val<uint32_t>(0);
    uint32_t src1Addr = get_arg_val<uint32_t>(1);
    uint32_t src2Addr = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    bool is_input1_sharded = get_arg_val<uint32_t>(5);
    bool is_input2_sharded = get_arg_val<uint32_t>(6);
    bool is_input3_sharded = get_arg_val<uint32_t>(7);
    uint32_t end_id = start_id + batch;
    const uint32_t single_tile_size_bytes = get_tile_size(src0_cb_index);
    // DPRINT << "START" << start_id << ENDL();
    // DPRINT << "BATCH" << batch << ENDL();
    if (is_input1_sharded) {
        cb_reserve_back(src0_cb_index, batch);
        // print_full_tile(src0_cb_index, 0);
        cb_push_back(src0_cb_index, batch);
    } else {
        const InterleavedAddrGenFast<true> a = {
            .bank_base_address = src0Addr,
            .page_size = single_tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_reserve_back(src0_cb_index, 1);
            DPRINT << "INPUT 1" << ENDL();

            uint32_t l1_write_addr_in0 = get_write_ptr(src0_cb_index);
            noc_async_read_tile(i, a, l1_write_addr_in0);
            noc_async_read_barrier();
            print_full_tile(src0_cb_index, 0);
            cb_push_back(src0_cb_index, 1);
        }
    }

    if (is_input2_sharded) {
        cb_reserve_back(src1_cb_index, batch);
        cb_push_back(src1_cb_index, batch);
    } else {
        const InterleavedAddrGenFast<true> b = {
            .bank_base_address = src1Addr,
            .page_size = single_tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_reserve_back(src1_cb_index, 1);
            DPRINT << "INPUT 2" << ENDL();
            uint32_t l1_write_addr_in1 = get_write_ptr(src1_cb_index);
            noc_async_read_tile(i, b, l1_write_addr_in1);
            noc_async_read_barrier();

            print_full_tile(src1_cb_index, 0);
            cb_push_back(src1_cb_index, 1);
        }
    }

    if (is_input3_sharded) {
        cb_reserve_back(src2_cb_index, batch);
        DPRINT << "INPUT 3" << ENDL();
        print_full_tile(src2_cb_index, 0);
        cb_push_back(src2_cb_index, batch);
    } else {
        const InterleavedAddrGenFast<true> c = {
            .bank_base_address = src2Addr,
            .page_size = single_tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_reserve_back(src2_cb_index, 1);
            // DPRINT << "INPUT 3"<< ENDL();
            uint32_t l1_write_addr_in2 = get_write_ptr(src2_cb_index);
            noc_async_read_tile(i, c, l1_write_addr_in2);
            noc_async_read_barrier();
            // print_full_tile(src2_cb_index, 0);

            cb_push_back(src2_cb_index, 1);
        }
    }
}
