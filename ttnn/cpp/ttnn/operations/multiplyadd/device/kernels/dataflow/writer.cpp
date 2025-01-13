#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA1({ DPRINT << r << " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    uint8_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t dstAddr = get_arg_val<uint32_t>(0);
    uint32_t batch = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    bool is_output_sharded = get_arg_val<uint32_t>(3);
    uint32_t end_id = start_id + batch;
    const uint32_t single_tile_size_bytes = get_tile_size(dst1_cb_index);

    if (is_output_sharded) {
        // DPRINT << "SHARDED" << ENDL();
        // DPRINT << "START" << start_id << ENDL();
        cb_wait_front(dst1_cb_index, batch);
        cb_pop_front(dst1_cb_index, batch);
    } else {
        DPRINT << "DRAM INTERLEAVED" << ENDL();
        const InterleavedAddrGenFast<true> c = {
            .bank_base_address = dstAddr,
            .page_size = single_tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };
        // DPRINT << "STRAT" << start_id << ENDL();
        // DPRINT << "BATCH" << batch << ENDL();
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_wait_front(dst1_cb_index, 1);
            // DPRINT << "OUTPUT" << ENDL();
            // print_full_tile(dst1_cb_index, 0);
            uint32_t l1_read_addr = get_read_ptr(dst1_cb_index);
            noc_async_write_tile(i, c, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(dst1_cb_index, 1);
        }
    }
}
