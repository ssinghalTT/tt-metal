#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
// #include "debug/dprint.h"

using namespace tt;

void kernel_main() {
    // DPRINT << "WRITER" << ENDL();
    uint8_t dst1_cb_index = tt::CBIndex::c_4;
    uint32_t dstAddr = get_arg_val<uint32_t>(0);
    uint32_t batch = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t single_tile_size_bytes = get_tile_size(dst1_cb_index);

    const InterleavedAddrGenFast<true> c = {
        .bank_base_address = dstAddr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // DPRINT << "WRITER BATCH" << batch << ENDL();
    uint32_t end_id = start_id + batch;
    for (uint32_t i = start_id; i < end_id; i++) {
        cb_wait_front(dst1_cb_index, 1);
        uint32_t l1_read_addr = get_read_ptr(dst1_cb_index);
        // DPRINT_DATA1({ DPRINT << (uint)i << " DST " << TileSlice(4, 0, SliceRange::hw0_32_16(), dst1_cb_index,
        // l1_read_addr, true, false) << ENDL(); });
        noc_async_write_tile(i, c, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(dst1_cb_index, 1);
    }
    // DPRINT << "WRITER END" << ENDL();
}
